"""Typed immutable tabular inputs backed by individually stored tensors."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from exaflow.algorithms.feta.tensor import FedTensor


@dataclass(frozen=True)
class ColumnMetadata:
    source_type: str
    is_categorical: bool
    categories: tuple[str | int | float, ...] = ()
    labels: tuple[str, ...] = ()


class FedColumns:
    def __init__(
        self, columns: Mapping[str, FedTensor], metadata: Mapping[str, ColumnMetadata]
    ):
        if not columns or set(columns) != set(metadata):
            raise ValueError("FedColumns requires matching tensors and metadata.")
        first = next(iter(columns.values()))
        for tensor in columns.values():
            if tensor.partition_axis != 0 or tensor.shape != (first.shape[0], 1):
                raise ValueError("FedColumns requires aligned single-column tensors.")
            if tensor.runtime is not first.runtime or set(tensor.descriptors) != set(
                first.descriptors
            ):
                raise ValueError("FedColumns cannot mix execution contexts or workers.")
            for worker, descriptor in tensor.descriptors.items():
                reference = first.descriptors[worker]
                if (
                    descriptor.shape[0] != reference.shape[0]
                    or descriptor.row_layout != reference.row_layout
                ):
                    raise ValueError("FedColumns contains unaligned local rows.")
        self._columns = MappingProxyType(dict(columns))
        self._metadata = MappingProxyType(dict(metadata))
        self._names = tuple(columns)
        self._nrows = first.shape[0]

    def __getitem__(self, name: str) -> FedTensor:
        return self._columns[name]

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    @property
    def metadata(self) -> Mapping[str, ColumnMetadata]:
        return self._metadata

    @property
    def nrows(self) -> int:
        return self._nrows
