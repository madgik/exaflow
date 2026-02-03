from typing import Protocol
from typing import Sequence
from typing import Union

import numpy as np

ArrayInput = Union[
    Sequence[object],
    Sequence[Sequence[object]],
    np.ndarray,
]


class AggregationClient(Protocol):
    def sum(self, values: ArrayInput) -> np.ndarray: ...

    def min(self, values: ArrayInput) -> np.ndarray: ...

    def max(self, values: ArrayInput) -> np.ndarray: ...

    def union(self, values: Sequence[object]) -> Sequence[object]: ...
