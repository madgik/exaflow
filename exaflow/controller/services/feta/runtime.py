import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from threading import Lock
from threading import RLock
from uuid import uuid4

import numpy as np
from google.protobuf.json_format import ParseDict

from exaflow.algorithms.feta.tensor import FedTensor
from exaflow.feta.lifecycle import ContextClosedError
from exaflow.feta.operations import LocalOperation
from exaflow.feta.operations import LocalTensorRef
from exaflow.feta.operations import OperationType
from exaflow.feta.operations import TensorMetadata
from exaflow.feta.operations import aggregation_kind
from exaflow.feta.operations import infer_operation
from exaflow.feta.operations import validate_array
from exaflow.feta.settings import FetaSettings
from exaflow.feta.tensor import RuntimeFedTensor
from exaflow.protos.feta import feta_pb2 as pb
from exaflow.worker_communication import InsufficientDataError

LOGGER = logging.getLogger(__name__)


class FetaRuntime:
    def __init__(self, workers, aggregation, settings=None, context_id=None):
        self.all_workers = list(workers)
        self.workers = {}
        self.aggregation = aggregation
        self.settings = settings or FetaSettings()
        self.context_id = context_id or uuid4().hex
        self.deadline = time.monotonic() + self.settings.context_lifetime_minutes * 60
        self.step = 0
        self.closed = False
        self.execution_lock = RLock()
        self.state_lock = Lock()

    def initialize(
        self, inputdata, columns: dict[str, list[str]]
    ) -> dict[str, FedTensor]:
        if self.closed or self.workers:
            raise RuntimeError(
                "FETA runtime cannot be initialized twice or after cleanup."
            )
        if (
            not columns
            or set(columns) - {"x", "y"}
            or any(not group for group in columns.values())
        ):
            raise ValueError("FETA initialization requires selected x or y groups.")
        selection = pb.DataSelection(
            data_model=inputdata.data_model,
            datasets=inputdata.datasets,
            variables=inputdata.variables,
        )
        if inputdata.filters:
            ParseDict(inputdata.filters, selection.filters)
        request = pb.InitializeRequest(
            context_id=self.context_id, data=selection, **columns
        )
        responses = self._parallel(
            {w.worker_id: (w, request) for w in self.all_workers}, "initialize", True
        )
        if not responses:
            raise InsufficientDataError("No workers had sufficient data for FETA.")
        self.workers = {
            w.worker_id: w for w in self.all_workers if w.worker_id in responses
        }
        for worker_id, response in responses.items():
            if response.worker_id != worker_id:
                raise ValueError("FETA initialization worker identity mismatch.")
            returned_groups = {name for name in ("x", "y") if response.HasField(name)}
            if returned_groups != set(columns):
                raise ValueError(
                    "FETA initialization returned unexpected input groups."
                )
            row_counts = set()
            for name, selected in columns.items():
                descriptor = getattr(response, name)
                if (
                    descriptor.row_layout != self.context_id
                    or descriptor.partition_axis != 0
                    or not descriptor.tensor_id
                    or (
                        descriptor.dtype != "float64"
                        and np.dtype(descriptor.dtype).kind != "U"
                    )
                    or len(descriptor.shape) != 2
                    or descriptor.shape[0] < 0
                    or descriptor.shape[1] != len(selected)
                ):
                    raise ValueError("Invalid FETA input partition metadata.")
                row_counts.add(descriptor.shape[0])
            if len(row_counts) != 1:
                raise ValueError("FETA input groups have unaligned row counts.")
        return {
            name: RuntimeFedTensor(
                self, {key: getattr(value, name) for key, value in responses.items()}
            )
            for name in columns
        }

    def _parallel(self, calls, method, skip_insufficient=False):
        if not calls:
            return {}
        pool = ThreadPoolExecutor(max_workers=len(calls))
        futures = {
            pool.submit(getattr(worker, method), request): worker_id
            for worker_id, (worker, request) in calls.items()
        }
        results = {}
        try:
            for future in as_completed(futures):
                worker_id = futures[future]
                try:
                    results[worker_id] = future.result()
                except InsufficientDataError:
                    if not skip_insufficient:
                        raise
                    LOGGER.info("Skipping insufficient-data FETA worker %s.", worker_id)
                    calls[worker_id][0].cleanup(self.context_id)
            pool.shutdown(wait=True)
            return results
        except Exception:
            for future in futures:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            raise

    def apply(self, op_type, inputs, axis=None):
        with self.execution_lock:
            self.check_active()
            if op_type in (OperationType.MIN, OperationType.MAX):
                shape = inputs[0].shape
                if (axis is None and not np.prod(shape)) or (
                    axis is not None and 0 <= axis < len(shape) and shape[axis] == 0
                ):
                    raise ValueError("Zero-size array reduction has no identity.")
            calls, predictions = {}, {}
            for worker_id, worker in self.workers.items():
                operands, metadata = [], []
                for value in inputs:
                    if isinstance(value, FedTensor):
                        if value.runtime is not self or set(value.descriptors) != set(
                            self.workers
                        ):
                            raise ValueError(
                                "Cannot mix FETA tensors from different contexts."
                            )
                        descriptor = value.descriptors[worker_id]
                        operands.append(LocalTensorRef(descriptor.tensor_id))
                        metadata.append(TensorMetadata.from_descriptor(descriptor))
                    else:
                        array = validate_array(value, self.settings.max_tensor_bytes)
                        if not isinstance(
                            value,
                            (
                                np.ndarray,
                                int,
                                float,
                                str,
                                bool,
                                np.integer,
                                np.floating,
                                np.str_,
                                np.bool_,
                            ),
                        ):
                            raise ValueError("Unsupported FETA literal operand.")
                        operands.append(value)
                        metadata.append(
                            TensorMetadata(array.shape, str(array.dtype), -1, "")
                        )
                selection_layout = ""
                if op_type == OperationType.FILTER:
                    mask = inputs[1]
                    if not hasattr(mask, "selection_layout"):
                        mask.selection_layout = uuid4().hex
                    selection_layout = mask.selection_layout
                operation = LocalOperation(
                    op_type, tuple(operands), axis, selection_layout
                )
                predicted = infer_operation(
                    operation, metadata, self.settings.max_tensor_bytes
                )
                predictions[worker_id] = predicted
                calls[worker_id] = (
                    worker,
                    pb.ExecuteRequest(
                        context_id=self.context_id, operation=operation.to_proto()
                    ),
                )
            first = next(iter(predictions.values()))
            global_result = first.partition_axis == -1
            unique = op_type == OperationType.UNIQUE
            if global_result:
                if not unique and any(m != first for m in predictions.values()):
                    raise ValueError(
                        "FETA global result metadata differs across workers."
                    )
                self.step += 1
                self.aggregation.prepare(
                    pb.PrepareRequest(
                        context_id=self.context_id,
                        step=self.step,
                        worker_ids=sorted(self.workers),
                        shape=(sum(m.shape[0] for m in predictions.values()),)
                        if unique
                        else first.shape,
                        dtype=str(
                            np.result_type(
                                *[np.dtype(m.dtype) for m in predictions.values()]
                            )
                        )
                        if unique
                        else first.dtype,
                        kind=aggregation_kind(op_type),
                    )
                )
                for _, request in calls.values():
                    request.aggregation_step = self.step
            if global_result:
                result = self._execute_global(calls, predictions, unique)
                if unique:
                    max_count = sum(m.shape[0] for m in predictions.values())
                    dtype = np.result_type(
                        *[np.dtype(m.dtype) for m in predictions.values()]
                    )
                    if (
                        result.ndim != 1
                        or len(result) > max_count
                        or result.dtype != dtype
                    ):
                        raise ValueError(
                            "Aggregation returned invalid distinct-value metadata."
                        )
                    validate_array(result, self.settings.max_tensor_bytes)
                    return result
                if result.shape != first.shape or str(result.dtype) != first.dtype:
                    raise ValueError("Aggregation returned inconsistent FETA metadata.")
                return result.item() if result.ndim == 0 else result
            responses = self._parallel(calls, "execute")
            for worker_id, descriptor in responses.items():
                actual = TensorMetadata.from_descriptor(descriptor)
                predicted = predictions[worker_id]
                if op_type == OperationType.FILTER:
                    if (
                        len(actual.shape) != len(predicted.shape)
                        or actual.shape[1:] != predicted.shape[1:]
                        or not 0 <= actual.shape[0] <= predicted.shape[0]
                    ):
                        raise ValueError("Worker returned invalid filtered shape.")
                    predicted = TensorMetadata(
                        actual.shape, predicted.dtype, 0, predicted.row_layout
                    )
                if actual != predicted:
                    raise ValueError("Worker returned inconsistent FETA metadata.")
            return RuntimeFedTensor(self, responses)

    def check_active(self):
        if self.closed or not self.workers:
            raise ContextClosedError("FETA runtime is not active.")
        if time.monotonic() >= self.deadline:
            raise ContextClosedError("FETA execution context expired.")

    def _execute_global(self, calls, predictions, unique=False):
        pool = ThreadPoolExecutor(max_workers=len(calls) + 1)
        futures = {
            pool.submit(worker.execute, request): worker_id
            for worker_id, (worker, request) in calls.items()
        }
        fetch = pool.submit(self.aggregation.fetch, self.context_id, self.step)
        futures[fetch] = None
        try:
            result = None
            for future in as_completed(futures):
                worker_id = futures[future]
                value = future.result()
                if worker_id is None:
                    result = value
                else:
                    actual = TensorMetadata.from_descriptor(value)
                    predicted = predictions[worker_id]
                    if unique:
                        if (
                            len(actual.shape) != 1
                            or not 0 <= actual.shape[0] <= predicted.shape[0]
                        ):
                            raise ValueError(
                                "Worker returned invalid distinct-value shape."
                            )
                        predicted = TensorMetadata(
                            actual.shape, predicted.dtype, -1, predicted.row_layout
                        )
                    if actual != predicted:
                        raise ValueError("Worker returned inconsistent FETA metadata.")
            pool.shutdown(wait=True)
            return result
        except Exception:
            try:
                self.aggregation.cleanup(self.context_id)
            except Exception:
                LOGGER.exception(
                    "Failed to cancel FETA aggregation %s.", self.context_id
                )
            for future in futures:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            raise

    def close(self, suppress_errors=False):
        with self.state_lock:
            if self.closed:
                return
            self.closed = True
        errors = []
        for client in [self.aggregation, *self.all_workers]:
            client.cancel()
        for client in [self.aggregation, *self.all_workers]:
            try:
                client.cleanup(self.context_id)
            except Exception as exc:
                errors.append(exc)
                LOGGER.exception("Failed to clean up FETA context %s.", self.context_id)
            finally:
                client.close()
        if errors and not suppress_errors:
            raise errors[0]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close(suppress_errors=exc_type is not None)
