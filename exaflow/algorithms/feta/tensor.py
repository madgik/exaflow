from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Union
from typing import overload

import numpy as np
from numpy.typing import NDArray

TensorOperand = Union["FedTensor", NDArray, int, float, np.integer, np.floating]
TensorResult = Union["FedTensor", NDArray, int, float]


class FedTensor(ABC):
    """Typed algebra for distributed numerical arrays.

    Operations execute immediately. Results retaining a partitioned dimension
    are tensors; contracting it returns an aggregated NumPy array or scalar.
    Raw worker-local arrays cannot be retrieved through this interface.
    """

    @abstractmethod
    def __eq__(self, other: FedTensor | str | int | float | bool) -> FedTensor:
        """Compare aligned local values, returning a boolean tensor."""

    @abstractmethod
    def __getitem__(self, key: FedTensor | tuple[slice, int]) -> FedTensor:
        """Select rows with a boolean tensor, or a column with [:, index]."""

    def __bool__(self) -> bool:
        raise TypeError("The truth value of a FedTensor is ambiguous.")

    @abstractmethod
    def __add__(self, other: TensorOperand) -> FedTensor:
        """Apply the supported numerical operation with a tensor or literal."""

    @abstractmethod
    def __radd__(self, other: TensorOperand) -> FedTensor:
        """Apply the supported numerical operation with a tensor or literal."""

    @abstractmethod
    def __sub__(self, other: TensorOperand) -> FedTensor:
        """Apply the supported numerical operation with a tensor or literal."""

    @abstractmethod
    def __rsub__(self, other: TensorOperand) -> FedTensor:
        """Apply the supported numerical operation with a tensor or literal."""

    @abstractmethod
    def __mul__(self, other: TensorOperand) -> FedTensor:
        """Apply the supported numerical operation with a tensor or literal."""

    @abstractmethod
    def __rmul__(self, other: TensorOperand) -> FedTensor:
        """Apply the supported numerical operation with a tensor or literal."""

    @abstractmethod
    def __truediv__(self, other: TensorOperand) -> FedTensor:
        """Apply the supported numerical operation with a tensor or literal."""

    @abstractmethod
    def __rtruediv__(self, other: TensorOperand) -> FedTensor:
        """Apply the supported numerical operation with a tensor or literal."""

    @abstractmethod
    def __pow__(self, other: TensorOperand) -> FedTensor:
        """Apply the supported numerical operation with a tensor or literal."""

    @abstractmethod
    def __rpow__(self, other: TensorOperand) -> FedTensor:
        """Apply the supported numerical operation with a tensor or literal."""

    @abstractmethod
    def __matmul__(self, other: TensorOperand) -> FedTensor | NDArray:
        """Apply the supported numerical operation with a tensor or literal."""

    @abstractmethod
    def __rmatmul__(self, other: TensorOperand) -> FedTensor | NDArray:
        """Apply the supported numerical operation with a tensor or literal."""

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Global shape, computed from worker metadata without aggregation."""

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """NumPy numerical dtype shared by all partitions."""

    @property
    @abstractmethod
    def partition_axis(self) -> int:
        """Numeric axis along which the value is distributed."""

    @property
    @abstractmethod
    def T(self) -> FedTensor:
        """Transpose locally, moving the partitioned axis."""

    @overload
    def sum(self, axis: None = None) -> int | float: ...

    @overload
    def sum(self, axis: int) -> TensorResult: ...

    @abstractmethod
    def sum(self, axis: int | None = None) -> TensorResult:
        """Reduce axis 0 or 1; aggregate when reducing the partitioned axis."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the global first dimension, aggregating row-partitioned lengths."""
