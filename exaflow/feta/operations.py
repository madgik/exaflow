import math
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from exaflow.aggregation_serialization import bytes_to_ndarray
from exaflow.aggregation_serialization import ndarray_to_bytes
from exaflow.feta.lifecycle import ResourceLimitError
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
    if array.ndim > 2 or array.dtype.kind not in "iuf":
        raise ValueError("FETA supports numerical tensors of rank at most two.")
    if array.dtype.itemsize > 8 or (
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


@dataclass(frozen=True)
class LocalOperation:
    type: OperationType
    inputs: tuple
    axis: int | None = None

    def to_proto(self):
        message = pb.LocalOperation(type=int(self.type))
        if self.axis is not None:
            message.axis = self.axis
        for value in self.inputs:
            operand = message.inputs.add()
            if isinstance(value, LocalTensorRef):
                operand.tensor_id = value.tensor_id
            elif isinstance(value, (int, np.integer)) and not isinstance(value, bool):
                operand.integer = int(value)
            elif isinstance(value, (float, np.floating)):
                operand.real = float(value)
            elif isinstance(value, np.ndarray):
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
            2 if op_type in BINARY or op_type == OperationType.MATMUL else 1
        )
        if len(message.inputs) != expected_inputs:
            raise ValueError("Incorrect number of FETA operation inputs.")
        if message.HasField("axis") and op_type != OperationType.SUM:
            raise ValueError("Only SUM accepts an axis.")
        inputs = []
        for operand in message.inputs:
            kind = operand.WhichOneof("value")
            if kind == "tensor_id":
                if not operand.tensor_id or len(operand.tensor_id) > 128:
                    raise ValueError("Invalid FETA tensor reference.")
                inputs.append(LocalTensorRef(operand.tensor_id))
            elif kind == "array":
                inputs.append(decode_array(operand.array.arrow, max_bytes))
            elif kind in ("real", "integer"):
                inputs.append(getattr(operand, kind))
            else:
                raise ValueError("Missing FETA operand value.")
        axis = message.axis if message.HasField("axis") else None
        return cls(op_type, tuple(inputs), axis)


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
    expected = 2 if op in BINARY or op == OperationType.MATMUL else 1
    if len(metadata) != expected:
        raise ValueError("Incorrect number of FETA operation inputs.")
    if operation.axis is not None and op != OperationType.SUM:
        raise ValueError("Only SUM accepts an axis.")
    distributed = [m for m in metadata if m.partition_axis >= 0]
    if not distributed:
        raise ValueError("A local operation requires a distributed tensor input.")
    if any(m.row_layout != distributed[0].row_layout for m in distributed):
        raise ValueError("FETA operands have unaligned row partitions.")
    layout = distributed[0].row_layout
    dtype = np.result_type(*[np.dtype(m.dtype) for m in metadata])
    if op in BINARY:
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
        if op == OperationType.DIVIDE:
            dtype = np.result_type(dtype, np.float64)
    elif op == OperationType.TRANSPOSE:
        m = metadata[0]
        shape = m.shape[::-1]
        partition = len(m.shape) - 1 - m.partition_axis
    elif op == OperationType.SUM:
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
        dtype = np.asarray(np.sum(np.zeros((), dtype=dtype))).dtype
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
    elif op == OperationType.TRANSPOSE:
        result = inputs[0].T
    elif op == OperationType.SUM:
        result = np.sum(inputs[0], axis=operation.axis)
    elif op == OperationType.LENGTH:
        result = np.int64(len(inputs[0]))
    elif op == OperationType.MATMUL:
        result = inputs[0] @ inputs[1]
    else:
        raise ValueError("Unsupported FETA operation type.")
    # Canonical promotion is shared by controller and worker (inputs are float64).
    return np.asarray(result, dtype=expected.dtype)
