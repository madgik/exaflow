import numbers
import re

import numpy as np
import pyarrow as pa
from google.protobuf.json_format import MessageToDict

from exaflow.feta.clients import FetaAggregationClient
from exaflow.feta.inputdata import Inputdata
from exaflow.feta.lifecycle import ContextClosedError
from exaflow.feta.lifecycle import ContextStore
from exaflow.feta.lifecycle import ResourceLimitError
from exaflow.feta.operations import LocalOperation
from exaflow.feta.operations import LocalTensorRef
from exaflow.feta.operations import OperationType
from exaflow.feta.operations import TensorMetadata
from exaflow.feta.operations import encode_tensor
from exaflow.feta.operations import evaluate
from exaflow.feta.operations import infer_operation
from exaflow.feta.operations import validate_array
from exaflow.feta.rpc import abort
from exaflow.protos.feta import feta_pb2 as pb
from exaflow.protos.feta import feta_pb2_grpc as rpc
from exaflow.worker.feta.data_loader import load_algorithm_arrow_table
from exaflow.worker.feta.filters import FILTER_OPERATORS
from exaflow.worker_communication import InsufficientDataError


def validate_filter_wire(rule):
    """Guard the reused SQL builder even for requests bypassing the HTTP API."""
    if not rule:
        return
    if "condition" in rule:
        if rule["condition"] not in ("AND", "OR"):
            raise ValueError("Invalid FETA filter condition.")
        for child in rule.get("rules", []):
            validate_filter_wire(child)
        return
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", rule.get("id", "")):
        raise ValueError("Invalid FETA filter variable identifier.")
    if rule.get("operator") not in FILTER_OPERATORS:
        raise ValueError("Unsupported FETA filter operator.")
    if rule["operator"] in ("is_null", "is_not_null"):
        return
    values = rule.get("value")
    values = values if isinstance(values, list) else [values]
    if rule.get("type") == "string":
        if any(not isinstance(value, str) for value in values):
            raise ValueError("String filter requires string values.")
    elif any(not isinstance(value, numbers.Real) for value in values):
        raise ValueError("Numerical filter requires numerical values.")


class FetaWorkerService(rpc.FetaWorkerServicer):
    def __init__(
        self,
        worker_id,
        settings=None,
        minimum_rows=10,
        loader=load_algorithm_arrow_table,
        aggregation_address=None,
        aggregation_factory=FetaAggregationClient,
    ):
        self.worker_id = worker_id
        self.store = ContextStore(settings)
        self.minimum_rows = minimum_rows
        self.loader = loader
        self.aggregation_address = aggregation_address
        self.aggregation_factory = aggregation_factory

    def initialize(self, request, active=lambda: True):
        if not active():
            raise ContextClosedError("FETA initialization was cancelled.")
        data = request.data
        if not re.fullmatch(r"[A-Za-z0-9_.:-]+", data.data_model):
            raise ValueError("Invalid FETA data-model identifier.")
        if not data.datasets or not (request.x or request.y):
            raise ValueError("FETA requires datasets and a selected input group.")
        if len(set(request.column_groups)) != len(request.column_groups) or set(
            request.column_groups
        ) - {"x", "y"}:
            raise ValueError("Invalid FETA column groups.")
        if any(not getattr(request, group) for group in request.column_groups):
            raise ValueError("Column groups must have selected inputs.")
        variables = list(dict.fromkeys([*request.x, *request.y]))
        if not set(variables).issubset(data.variables):
            raise ValueError("FETA input columns are outside the selected variables.")
        if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) for name in variables):
            raise ValueError("Invalid FETA input variable identifier.")
        filters = MessageToDict(data.filters) if data.HasField("filters") else None
        validate_filter_wire(filters)
        inputdata = Inputdata(
            data_model=data.data_model,
            datasets=list(data.datasets),
            variables=variables,
            filters=filters,
        )
        # Reserve before loading: cleanup can close initialization in flight.
        self.store.create(request.context_id, {})
        ctx, _ = self.store.acquire(request.context_id, [])
        try:
            return self._load_inputs(request, inputdata, variables, ctx, active)
        except Exception:
            self.store.cleanup(request.context_id, expected_context=ctx)
            raise

    def _load_inputs(self, request, inputdata, variables, ctx, active):
        if not active():
            raise ContextClosedError("FETA initialization was cancelled.")
        table = self.loader(inputdata, include_dataset=False)
        nrows = getattr(table, "num_rows", len(table))
        if nrows < self.minimum_rows:
            raise InsufficientDataError(
                f"Insufficient data returned {nrows} rows; minimum required is "
                f"{self.minimum_rows}."
            )
        if getattr(table, "nbytes", 0) > self.store.settings.max_context_bytes:
            raise ResourceLimitError("FETA input exceeds the context memory limit.")
        groups = {
            name: tuple(columns)
            for name, columns in (("x", request.x), ("y", request.y))
            if columns
        }
        selections = {
            (selection, group in request.column_groups)
            for group, names in groups.items()
            for selection in (
                [(name,) for name in names]
                if group in request.column_groups
                else [names]
            )
        }
        frame = table.to_pandas()
        # TODO: replace temporary missing-value rejection when preprocessing is designed.
        if frame[variables].isna().any().any():
            raise ValueError("FETA does not yet support missing input values.")
        tensors = {}
        for columns, individual in selections:
            selected = frame[list(columns)]
            text_columns = [
                pa.types.is_string(table.schema.field(c).type)
                or pa.types.is_large_string(table.schema.field(c).type)
                or (
                    pa.types.is_null(table.schema.field(c).type)
                    and selected[c].dtype.kind == "O"
                )
                for c in columns
            ]
            if any(text_columns) and not all(text_columns):
                raise ValueError(
                    "FETA input groups cannot mix text and numerical columns."
                )
            remaining = self.store.settings.max_context_bytes - sum(
                a.nbytes for a, _ in tensors.values()
            )
            if all(text_columns):
                # Fixed-width Unicode avoids arbitrary Python objects in the store.
                width = max(
                    1, max((len(v) for c in columns for v in selected[c]), default=1)
                )
                predicted_bytes = nrows * len(columns) * width * 4
                if predicted_bytes > min(
                    self.store.settings.max_tensor_bytes, remaining
                ):
                    raise ResourceLimitError(
                        "FETA input exceeds the tensor memory limit."
                    )
                array = selected.to_numpy(dtype=f"U{width}")
            else:
                predicted_bytes = nrows * len(columns) * 8
                if predicted_bytes > min(
                    self.store.settings.max_tensor_bytes, remaining
                ):
                    raise ResourceLimitError(
                        "FETA input exceeds the tensor memory limit."
                    )
                preserve_integer = (
                    len(columns) == 1
                    and individual
                    and pa.types.is_integer(table.schema.field(columns[0]).type)
                )
                array = (
                    selected.to_numpy(
                        dtype=table.schema.field(columns[0]).type.to_pandas_dtype()
                    )
                    if preserve_integer
                    else selected.to_numpy(dtype=np.float64)
                )
            validate_array(array, self.store.settings.max_tensor_bytes)
            tensors[(columns, individual)] = (
                array,
                TensorMetadata(array.shape, str(array.dtype), 0, request.context_id),
            )
        if (
            sum(array.nbytes for array, _ in tensors.values())
            > self.store.settings.max_context_bytes
        ):
            raise ResourceLimitError("FETA input exceeds the context memory limit.")
        if not active():
            raise ContextClosedError("FETA initialization was cancelled.")
        refs = {
            columns: self.store.put(request.context_id, ctx, array, metadata)
            for columns, (array, metadata) in tensors.items()
        }
        response = pb.InitializeResponse(worker_id=self.worker_id)
        for name, columns in groups.items():
            if name in request.column_groups:
                for column in columns:
                    response.columns.add(
                        group=name,
                        name=column,
                        tensor=tensors[((column,), True)][1].descriptor(
                            refs[((column,), True)]
                        ),
                    )
                continue
            getattr(response, name).CopyFrom(
                tensors[(columns, False)][1].descriptor(refs[(columns, False)])
            )
        return response

    def execute(self, request, active=lambda: True):
        if not active():
            raise ContextClosedError("FETA operation was cancelled.")
        operation = LocalOperation.from_proto(
            request.operation, self.store.settings.max_tensor_bytes
        )
        ids = [v.tensor_id for v in operation.inputs if isinstance(v, LocalTensorRef)]
        ctx, stored = self.store.acquire(request.context_id, ids)
        stored = iter(stored)
        inputs, metadata = [], []
        for operand in operation.inputs:
            if isinstance(operand, LocalTensorRef):
                array, meta = next(stored)
            else:
                array = validate_array(operand, self.store.settings.max_tensor_bytes)
                meta = TensorMetadata(array.shape, str(array.dtype), -1, "")
            inputs.append(array)
            metadata.append(meta)
        result_meta = infer_operation(
            operation, metadata, self.store.settings.max_tensor_bytes
        )
        global_result = result_meta.partition_axis == -1
        if global_result != bool(request.aggregation_step):
            raise ValueError(
                "Aggregation instruction differs from operation placement."
            )
        result = evaluate(operation, inputs, result_meta)
        if operation.type in (OperationType.FILTER, OperationType.UNIQUE):
            result_meta = TensorMetadata(
                result.shape,
                result_meta.dtype,
                result_meta.partition_axis,
                result_meta.row_layout,
            )
        if not active():
            raise ContextClosedError("FETA operation was cancelled.")
        tensor_id = self.store.put(request.context_id, ctx, result, result_meta)
        if global_result:
            self.store.check_active(request.context_id, ctx)
            client = self.aggregation_factory(self.aggregation_address)
            try:
                client.submit(
                    pb.SubmitRequest(
                        context_id=request.context_id,
                        step=request.aggregation_step,
                        worker_id=self.worker_id,
                        value=encode_tensor(result),
                    )
                )
            finally:
                client.close()
        return result_meta.descriptor(tensor_id)

    def InitializeContext(self, request, context):
        try:
            return self.initialize(request, context.is_active)
        except Exception as exc:
            abort(context, exc)

    def Execute(self, request, context):
        try:
            return self.execute(request, context.is_active)
        except Exception as exc:
            abort(context, exc)

    def Cleanup(self, request, context):
        self.store.cleanup(request.context_id)
        return pb.Ack()
