import math
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from exaflow.feta.lifecycle import ResourceLimitError
from exaflow.feta.serialization import bytes_to_ndarray
from exaflow.feta.serialization import ndarray_to_bytes
from exaflow.protos.feta import feta_pb2 as pb


class OperationType(IntEnum):
    ADD = pb.ADD
    SUBTRACT = pb.SUBTRACT
    MULTIPLY = pb.MULTIPLY
    DIVIDE = pb.DIVIDE
    POWER = pb.POWER
    TRANSPOSE = pb.TRANSPOSE
    SUM = pb.SUM
    MATMUL = pb.MATMUL
    LENGTH = pb.LENGTH
    EQUAL = pb.EQUAL
    COLUMN = pb.COLUMN
    FILTER = pb.FILTER
    MIN = pb.MIN
    MAX = pb.MAX
    UNIQUE = pb.UNIQUE
    ONES_LIKE = pb.ONES_LIKE
    COLUMN_STACK = pb.COLUMN_STACK
    EXP = pb.EXP
    LOG = pb.LOG
    WHERE = pb.WHERE


@dataclass(frozen=True)
class LocalTensorRef:
    tensor_id: str


@dataclass(frozen=True)
class TensorMetadata:
    shape: tuple
    dtype: str
    partition_axis: int
    row_layout: str

    def descriptor(self, tensor_id):
        return pb.TensorDescriptor(
            tensor_id=tensor_id,
            shape=self.shape,
            dtype=self.dtype,
            partition_axis=self.partition_axis,
            row_layout=self.row_layout,
        )

    @classmethod
    def from_descriptor(cls, descriptor):
        return cls(
            tuple(descriptor.shape),
            descriptor.dtype,
            descriptor.partition_axis,
            descriptor.row_layout,
        )


def validate_array(array, max_bytes):
    array = np.asarray(array)
    if array.ndim > 2 or array.dtype.kind not in "iufbU":
        raise ValueError(
            "FETA supports numeric, boolean and Unicode tensors of rank at most two."
        )
    if (array.dtype.kind in "iuf" and array.dtype.itemsize > 8) or (
        array.dtype.kind == "f" and array.dtype.itemsize < 4
    ):
        raise ValueError("Unsupported FETA numerical dtype.")
    if not array.dtype.isnative:
        raise ValueError("FETA requires native-endian numerical tensors.")
    if array.nbytes > max_bytes:
        raise ResourceLimitError("FETA tensor memory limit exceeded.")
    return array


def decode_array(payload, max_bytes):
    if len(payload) > max_bytes + 4096:
        raise ResourceLimitError("FETA tensor payload limit exceeded.")
    try:
        return validate_array(bytes_to_ndarray(payload), max_bytes)
    except (ValueError, ResourceLimitError):
        raise
    except Exception as exc:
        raise ValueError("Invalid Arrow tensor payload.") from exc


def encode_tensor(value):
    """Typed Unicode transport for distinct labels; numeric values use Arrow."""
    if value.dtype.kind == "U":
        if value.ndim != 1:
            raise ValueError("String transport requires a one-dimensional array.")
        return pb.TensorValue(
            strings=pb.StringArray(values=value.tolist(), dtype=str(value.dtype))
        )
    return pb.TensorValue(arrow=ndarray_to_bytes(value))


def decode_tensor(message, max_bytes):
    encoding = message.WhichOneof("encoding")
    if encoding == "arrow":
        return decode_array(message.arrow, max_bytes)
    if encoding != "strings":
        raise ValueError("Missing FETA tensor encoding.")
    try:
        dtype = np.dtype(message.strings.dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid FETA string array dtype.") from exc
    if dtype.kind != "U" or not dtype.isnative or not dtype.itemsize:
        raise ValueError("Invalid FETA string array dtype.")
    values = message.strings.values
    if len(values) * dtype.itemsize > max_bytes:
        raise ResourceLimitError("FETA string array exceeds the tensor limit.")
    if any(len(v) * 4 > dtype.itemsize for v in values):
        raise ValueError("String exceeds declared Unicode width.")
    return np.asarray(values, dtype=dtype)


def aggregation_kind(op):
    return {
        OperationType.MIN: pb.MIN_VALUES,
        OperationType.MAX: pb.MAX_VALUES,
        OperationType.UNIQUE: pb.UNION_VALUES,
    }.get(op, pb.SUM_VALUES)


@dataclass(frozen=True)
class LocalOperation:
    type: OperationType
    inputs: tuple
    axis: int | None = None
    selection_layout: str = ""

    def to_proto(self):
        message = pb.LocalOperation(
            type=int(self.type), selection_layout=self.selection_layout
        )
        if self.axis is not None:
            message.axis = self.axis
        for value in self.inputs:
            operand = message.inputs.add()
            if isinstance(value, LocalTensorRef):
                operand.tensor_id = value.tensor_id
            elif isinstance(value, (str, np.str_)):
                operand.text = str(value)
            elif isinstance(value, (bool, np.bool_)):
                operand.boolean = bool(value)
            elif isinstance(value, (int, np.integer)) and not isinstance(value, bool):
                operand.integer = int(value)
            elif isinstance(value, (float, np.floating)):
                operand.real = float(value)
            elif isinstance(value, np.ndarray):
                if value.dtype.kind == "U":
                    raise ValueError(
                        "String array literals are not yet supported; use string scalars."
                    )
                operand.array_dtype = str(value.dtype)
                operand.array.arrow = ndarray_to_bytes(value)
            else:
                raise ValueError("Unsupported FETA operand type.")
        return message

    @classmethod
    def from_proto(cls, message, max_bytes):
        try:
            op_type = OperationType(message.type)
        except ValueError as exc:
            raise ValueError("Unsupported FETA operation type.") from exc
        expected_inputs = (
            2
            if op_type in BINARY
            or op_type
            in (OperationType.MATMUL, OperationType.EQUAL, OperationType.FILTER)
            else 1
        )
        if op_type == OperationType.WHERE:
            expected_inputs = 3
        if op_type == OperationType.COLUMN_STACK:
            expected_inputs = len(message.inputs)
        if not message.inputs or len(message.inputs) != expected_inputs:
            raise ValueError("Incorrect number of FETA operation inputs.")
        if message.HasField("axis") and op_type not in (
            OperationType.SUM,
            OperationType.COLUMN,
            OperationType.MIN,
            OperationType.MAX,
        ):
            raise ValueError("Only reductions and COLUMN accept an axis.")
        inputs = []
        for operand in message.inputs:
            kind = operand.WhichOneof("value")
            if operand.array_dtype and kind != "array":
                raise ValueError("Only array operands accept a dtype.")
            if kind == "tensor_id":
                if not operand.tensor_id or len(operand.tensor_id) > 128:
                    raise ValueError("Invalid FETA tensor reference.")
                inputs.append(LocalTensorRef(operand.tensor_id))
            elif kind == "array":
                array = decode_array(operand.array.arrow, max_bytes)
                dtype = np.dtype(operand.array_dtype)
                if dtype == np.dtype("bool"):
                    if array.dtype != np.dtype("uint8") or np.any(array > 1):
                        raise ValueError("Invalid boolean array encoding.")
                    array = array.view(np.bool_)
                elif str(array.dtype) != operand.array_dtype:
                    raise ValueError("Array operand dtype differs from its encoding.")
                inputs.append(array)
            elif kind in ("real", "integer", "text", "boolean"):
                if kind == "text" and len(operand.text) * 4 > max_bytes:
                    raise ResourceLimitError(
                        "FETA string literal exceeds the tensor limit."
                    )
                inputs.append(getattr(operand, kind))
            else:
                raise ValueError("Missing FETA operand value.")
        axis = message.axis if message.HasField("axis") else None
        return cls(op_type, tuple(inputs), axis, message.selection_layout)


BINARY = {
    OperationType.ADD: np.add,
    OperationType.SUBTRACT: np.subtract,
    OperationType.MULTIPLY: np.multiply,
    OperationType.DIVIDE: np.divide,
    OperationType.POWER: np.power,
}


def infer_operation(operation, metadata, max_bytes):
    """Validate placement and predict allocation before evaluating NumPy."""
    op = operation.type
    expected = (
        2
        if op in BINARY
        or op in (OperationType.MATMUL, OperationType.EQUAL, OperationType.FILTER)
        else 1
    )
    if op == OperationType.WHERE:
        expected = 3
    if op == OperationType.COLUMN_STACK:
        expected = len(metadata)
    if not metadata or len(metadata) != expected:
        raise ValueError("Incorrect number of FETA operation inputs.")
    if operation.axis is not None and op not in (
        OperationType.SUM,
        OperationType.COLUMN,
        OperationType.MIN,
        OperationType.MAX,
    ):
        raise ValueError("Only reductions and COLUMN accept an axis.")
    distributed = [m for m in metadata if m.partition_axis >= 0]
    if not distributed:
        raise ValueError("A local operation requires a distributed tensor input.")
    if any(m.row_layout != distributed[0].row_layout for m in distributed):
        raise ValueError("FETA operands have unaligned row partitions.")
    layout = distributed[0].row_layout
    dtype = np.result_type(*[np.dtype(m.dtype) for m in metadata])
    if operation.selection_layout and op != OperationType.FILTER:
        raise ValueError("Only FILTER accepts selection lineage.")
    if op in BINARY or op in (OperationType.SUM, OperationType.MATMUL):
        if any(np.dtype(m.dtype).kind not in "iufb" for m in metadata):
            raise ValueError("Arithmetic and reductions require numerical tensors.")
    if op in BINARY or op in (OperationType.EQUAL, OperationType.WHERE):
        if (
            op == OperationType.EQUAL
            and len({np.dtype(m.dtype).kind == "U" for m in metadata}) != 1
        ):
            raise ValueError(
                "Equality operands must both be strings or both numerical."
            )
        shape = np.broadcast_shapes(*[m.shape for m in metadata])
        axes = [m.partition_axis + len(shape) - len(m.shape) for m in distributed]
        if len(set(axes)) != 1:
            raise ValueError("FETA broadcasting cannot change row alignment.")
        partition = axes[0]
        lengths = [m.shape[m.partition_axis] for m in distributed]
        if len(set(lengths)) != 1 or shape[partition] != lengths[0]:
            raise ValueError("Cannot broadcast across a partitioned dimension.")
        for m in metadata:
            if m.partition_axis < 0:
                padded = (1,) * (len(shape) - len(m.shape)) + m.shape
                if padded[partition] != 1:
                    raise ValueError("Literal operands must broadcast over records.")
        if op == OperationType.WHERE:
            if np.dtype(metadata[0].dtype).kind != "b" or any(
                np.dtype(m.dtype).kind not in "iufb" for m in metadata[1:]
            ):
                raise ValueError("WHERE requires a boolean mask and numerical values.")
            dtype = np.result_type(*[np.dtype(m.dtype) for m in metadata[1:]])
        if op == OperationType.EQUAL:
            dtype = np.dtype("bool")
        if op == OperationType.DIVIDE:
            dtype = np.result_type(dtype, np.float64)
    elif op in (OperationType.EXP, OperationType.LOG):
        m = metadata[0]
        if dtype.kind not in "iufb":
            raise ValueError("EXP and LOG require numerical tensors.")
        shape, partition = m.shape, m.partition_axis
        dtype = np.result_type(dtype, np.float64)
    elif op == OperationType.ONES_LIKE:
        m = metadata[0]
        if dtype.kind not in "iufb":
            raise ValueError("ONES_LIKE requires numerical tensors.")
        shape, partition = m.shape, m.partition_axis
    elif op == OperationType.COLUMN_STACK:
        if len({np.dtype(m.dtype).kind == "U" for m in metadata}) != 1:
            raise ValueError("COLUMN_STACK cannot mix text and numerical columns.")
        if any(m.partition_axis != 0 or len(m.shape) not in (1, 2) for m in metadata):
            raise ValueError(
                "COLUMN_STACK requires row-partitioned vectors or matrices."
            )
        if len({m.shape[0] for m in metadata}) != 1:
            raise ValueError("COLUMN_STACK requires matching local row counts.")
        shape = (
            metadata[0].shape[0],
            sum(1 if len(m.shape) == 1 else m.shape[1] for m in metadata),
        )
        partition = 0
    elif op == OperationType.COLUMN:
        m = metadata[0]
        if len(m.shape) != 2 or m.partition_axis != 0 or operation.axis is None:
            raise ValueError("Column indexing requires a row-partitioned matrix.")
        column = operation.axis
        if not -m.shape[1] <= column < m.shape[1]:
            raise ValueError("FETA column index out of bounds.")
        shape, partition = (m.shape[0],), 0
    elif op == OperationType.FILTER:
        data, mask = metadata
        if (
            data.partition_axis != 0
            or mask.partition_axis != 0
            or mask.shape != (data.shape[0],)
            or np.dtype(mask.dtype).kind != "b"
        ):
            raise ValueError(
                "Filtering requires an aligned one-dimensional boolean mask."
            )
        if not operation.selection_layout or len(operation.selection_layout) > 128:
            raise ValueError("Filtering requires valid selection lineage.")
        shape, partition, dtype = data.shape, 0, np.dtype(data.dtype)
        layout = operation.selection_layout
    elif op == OperationType.TRANSPOSE:
        m = metadata[0]
        shape = m.shape[::-1]
        partition = len(m.shape) - 1 - m.partition_axis
    elif op == OperationType.UNIQUE:
        if dtype.kind != "U":
            raise ValueError(
                "Distributed UNIQUE initially supports string category labels only."
            )
        shape, partition = (math.prod(metadata[0].shape),), -1
    elif op in (OperationType.SUM, OperationType.MIN, OperationType.MAX):
        m = metadata[0]
        axis = operation.axis
        if axis is not None and (axis < 0 or axis >= len(m.shape)):
            raise ValueError("Invalid FETA reduction axis.")
        shape = () if axis is None else m.shape[:axis] + m.shape[axis + 1 :]
        partition = (
            -1
            if axis is None or axis == m.partition_axis
            else (m.partition_axis - int(axis < m.partition_axis))
        )
        if op == OperationType.SUM:
            dtype = np.asarray(np.sum(np.zeros((), dtype=dtype))).dtype
        elif dtype.kind not in "iuf":
            raise ValueError("MIN and MAX require numerical tensors.")
        elif partition >= 0 and axis is not None and m.shape[axis] == 0:
            raise ValueError("Zero-size array reduction has no identity.")
    elif op == OperationType.LENGTH:
        m = metadata[0]
        if m.partition_axis != 0:
            raise ValueError("Federated LENGTH requires partitioning on axis zero.")
        shape, partition, dtype = (), -1, np.dtype("int64")
    elif op == OperationType.MATMUL:
        left, right = metadata
        if len(left.shape) != 2 or len(right.shape) != 2:
            raise ValueError("FETA matrix multiplication requires two matrices.")
        if left.shape[1] != right.shape[0]:
            raise ValueError("Incompatible FETA matrix dimensions.")
        shape = (left.shape[0], right.shape[1])
        if left.partition_axis == 1 and right.partition_axis == 0:
            partition = -1
        elif left.partition_axis == 0 and right.partition_axis == -1:
            partition = 0
        elif left.partition_axis == -1 and right.partition_axis == 1:
            partition = 1
        else:
            raise ValueError("Unsupported FETA matrix partitioning.")
    else:
        raise ValueError("Unsupported FETA operation type.")
    if len(shape) > 2 or math.prod(shape) * dtype.itemsize > max_bytes:
        raise ResourceLimitError("FETA operation result exceeds the tensor limit.")
    return TensorMetadata(tuple(shape), str(dtype), partition, layout)


def evaluate(operation, inputs, expected):
    op = operation.type
    if op in BINARY:
        result = BINARY[op](*inputs)
    elif op == OperationType.EQUAL:
        result = np.equal(*inputs)
    elif op == OperationType.EXP:
        result = np.exp(inputs[0])
    elif op == OperationType.LOG:
        result = np.log(inputs[0])
    elif op == OperationType.WHERE:
        result = np.where(*inputs)
    elif op == OperationType.ONES_LIKE:
        result = np.ones_like(inputs[0])
    elif op == OperationType.COLUMN_STACK:
        result = np.column_stack(inputs)
    elif op == OperationType.COLUMN:
        result = inputs[0][:, operation.axis]
    elif op == OperationType.FILTER:
        result = inputs[0][inputs[1]]
    elif op == OperationType.TRANSPOSE:
        result = inputs[0].T
    elif op == OperationType.SUM:
        result = np.sum(inputs[0], axis=operation.axis)
    elif op == OperationType.UNIQUE:
        result = np.unique(inputs[0])
    elif op in (OperationType.MIN, OperationType.MAX):
        dtype = np.dtype(expected.dtype)
        minimum = op == OperationType.MIN
        if dtype.kind == "f":
            neutral = np.inf if minimum else -np.inf
        else:
            limits = np.iinfo(dtype)
            neutral = limits.max if minimum else limits.min
        function = np.min if minimum else np.max
        result = function(inputs[0], axis=operation.axis, initial=neutral)
    elif op == OperationType.LENGTH:
        result = np.int64(len(inputs[0]))
    elif op == OperationType.MATMUL:
        result = inputs[0] @ inputs[1]
    else:
        raise ValueError("Unsupported FETA operation type.")
    # Canonical promotion is shared by controller and worker (inputs are float64).
    return np.asarray(result, dtype=expected.dtype)
