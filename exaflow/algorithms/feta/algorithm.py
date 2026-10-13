from abc import ABC
from abc import abstractmethod
from typing import ClassVar

from exaflow.algorithms.feta.columns import FedColumns
from exaflow.algorithms.feta.tensor import FedTensor


class Algorithm(ABC):
    column_inputs: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, *, parameters=None, x_names=(), y_names=()):
        self.parameters = dict(parameters or {})
        self.x_names = list(x_names)
        self.y_names = list(y_names)

    @classmethod
    @abstractmethod
    def get_specification(cls):
        """Return a specification with AlgorithmType.FETA."""

    @abstractmethod
    def run(self, **inputs: FedTensor | FedColumns):
        """Subclasses declare typed named inputs matching their specification."""
