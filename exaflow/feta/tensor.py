from __future__ import annotations

from typing import Literal
from typing import overload

import numpy as np
from numpy.typing import NDArray

from exaflow.algorithms.feta.tensor import FedTensor
from exaflow.algorithms.feta.tensor import TensorOperand
from exaflow.algorithms.feta.tensor import TensorResult
from exaflow.feta.operations import OperationType


class RuntimeFedTensor(FedTensor):
    """Controller-side references to aligned worker-local numerical tensors."""

    __array_priority__ = 1000

    def __init__(self, runtime, descriptors):
        self.runtime = runtime
        self.descriptors = descriptors

    @property
    def shape(self) -> tuple[int, ...]:
        descriptors = list(self.descriptors.values())
        dimensions = list(descriptors[0].shape)
        axis = self.partition_axis
        if axis >= 0:
            dimensions[axis] = sum(item.shape[axis] for item in descriptors)
        return tuple(dimensions)

    @property
    def dtype(self) -> np.dtype:
        return np.result_type(*[np.dtype(d.dtype) for d in self.descriptors.values()])

    @property
    def partition_axis(self) -> int:
        return next(iter(self.descriptors.values())).partition_axis

    def _binary(self, operation, other, reverse=False):
        inputs = (other, self) if reverse else (self, other)
        return self.runtime.apply(operation, inputs)

    def _reduce(
        self, operation: Literal["min", "max"], axis: int | None = None
    ) -> TensorResult:
        return self.runtime.apply(
            {"min": OperationType.MIN, "max": OperationType.MAX}[operation],
            (self,),
            axis,
        )

    def _unique(self) -> NDArray:
        return self.runtime.apply(OperationType.UNIQUE, (self,))

    def __eq__(self, other: FedTensor | str | int | float | bool) -> FedTensor:
        return self._binary(OperationType.EQUAL, other)

    def __getitem__(self, key: FedTensor | tuple[slice, int]) -> FedTensor:
        if isinstance(key, FedTensor):
            return self.runtime.apply(OperationType.FILTER, (self, key))
        if (
            isinstance(key, tuple)
            and len(key) == 2
            and isinstance(key[0], slice)
            and key[0] == slice(None)
            and isinstance(key[1], int)
            and not isinstance(key[1], bool)
        ):
            return self.runtime.apply(OperationType.COLUMN, (self,), key[1])
        raise TypeError("FETA indexing supports boolean row masks and [:, column].")

    def __add__(self, other: TensorOperand) -> FedTensor:
        return self._binary(OperationType.ADD, other)

    def __radd__(self, other: TensorOperand) -> FedTensor:
        return self._binary(OperationType.ADD, other, True)

    def __sub__(self, other: TensorOperand) -> FedTensor:
        return self._binary(OperationType.SUBTRACT, other)

    def __rsub__(self, other: TensorOperand) -> FedTensor:
        return self._binary(OperationType.SUBTRACT, other, True)

    def __mul__(self, other: TensorOperand) -> FedTensor:
        return self._binary(OperationType.MULTIPLY, other)

    def __rmul__(self, other: TensorOperand) -> FedTensor:
        return self._binary(OperationType.MULTIPLY, other, True)

    def __truediv__(self, other: TensorOperand) -> FedTensor:
        return self._binary(OperationType.DIVIDE, other)

    def __rtruediv__(self, other: TensorOperand) -> FedTensor:
        return self._binary(OperationType.DIVIDE, other, True)

    def __pow__(self, other: TensorOperand) -> FedTensor:
        return self._binary(OperationType.POWER, other)

    def __rpow__(self, other: TensorOperand) -> FedTensor:
        return self._binary(OperationType.POWER, other, True)

    def __matmul__(self, other: TensorOperand) -> FedTensor | NDArray:
        return self._binary(OperationType.MATMUL, other)

    def __rmatmul__(self, other: TensorOperand) -> FedTensor | NDArray:
        return self._binary(OperationType.MATMUL, other, True)

    @property
    def T(self) -> FedTensor:
        return self.runtime.apply(OperationType.TRANSPOSE, (self,))

    @overload
    def sum(self, axis: None = None) -> int | float: ...

    @overload
    def sum(self, axis: int) -> TensorResult: ...

    def sum(self, axis: int | None = None) -> TensorResult:
        return self.runtime.apply(OperationType.SUM, (self,), axis)

    def __len__(self) -> int:
        descriptor = next(iter(self.descriptors.values()))
        if descriptor.partition_axis != 0:
            return descriptor.shape[0]
        return int(self.runtime.apply(OperationType.LENGTH, (self,)))

    def __array__(self, dtype=None, copy=None):
        raise TypeError("A FedTensor cannot be converted to a local array.")

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        supported = {
            np.equal: OperationType.EQUAL,
            np.add: OperationType.ADD,
            np.subtract: OperationType.SUBTRACT,
            np.multiply: OperationType.MULTIPLY,
            np.divide: OperationType.DIVIDE,
            np.power: OperationType.POWER,
            np.matmul: OperationType.MATMUL,
        }
        if method != "__call__" or kwargs or ufunc not in supported:
            raise TypeError("Unsupported NumPy operation on FedTensor.")
        return self.runtime.apply(supported[ufunc], inputs)
