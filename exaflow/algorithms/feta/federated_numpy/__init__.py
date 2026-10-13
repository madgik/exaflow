"""Typed NumPy-style functions for FETA tensors and concrete numerical arrays."""

from __future__ import annotations

from typing import overload

import numpy as np
from numpy.typing import NDArray

from exaflow.algorithms.feta.tensor import FedTensor
from exaflow.algorithms.feta.tensor import TensorResult

__all__ = ["sum", "transpose", "equal"]


def _numerical_array(a: NDArray) -> NDArray:
    if not isinstance(a, np.ndarray) or a.dtype.kind not in "iufb":
        raise TypeError("Expected a FedTensor or a real numerical NumPy array.")
    return a


@overload
def sum(a: FedTensor, axis: None = None) -> int | float: ...


@overload
def sum(a: FedTensor, axis: int) -> TensorResult: ...


@overload
def sum(a: NDArray, axis: int | None = None) -> NDArray | np.integer | np.floating: ...


def sum(
    a: FedTensor | NDArray, axis: int | None = None
) -> TensorResult | np.integer | np.floating:
    """Sum values, aggregating when a distributed dimension is reduced.

    FedTensor inputs support the existing numeric axes 0 and 1, or None for
    a full reduction. Concrete real numerical arrays use NumPy's sum behavior.
    Only the axis argument is supported; dtype, out and keepdims are excluded.
    """
    if isinstance(a, FedTensor):
        return a.sum(axis=axis)
    return np.sum(_numerical_array(a), axis=axis)


@overload
def transpose(a: FedTensor) -> FedTensor: ...


@overload
def transpose(a: NDArray) -> NDArray: ...


def transpose(a: FedTensor | NDArray) -> FedTensor | NDArray:
    """Reverse the axes, preserving distributed data and moving its partition.

    This is the function form of .T. Custom axis permutations are not supported.
    """
    if isinstance(a, FedTensor):
        return a.T
    if not isinstance(a, np.ndarray) or a.dtype.kind not in "iufbU":
        raise TypeError(
            "Expected a FedTensor or a numerical NumPy array or Unicode array."
        )
    return np.transpose(a)


@overload
def equal(a: FedTensor, b: FedTensor | str | int | float | bool) -> FedTensor: ...


@overload
def equal(a: NDArray, b: NDArray | str | int | float | bool) -> NDArray: ...


def equal(a, b):
    """Elementwise equality, retaining the distributed row layout."""
    if isinstance(a, FedTensor):
        return a == b
    if isinstance(b, FedTensor):
        return b == a
    return np.equal(a, b)
