from abc import ABC
from abc import abstractmethod

from exaflow.algorithms.feta.tensor import FedTensor


class Algorithm(ABC):
    def __init__(self, *, parameters=None, x_names=(), y_names=()):
        self.parameters = dict(parameters or {})
        self.x_names = list(x_names)
        self.y_names = list(y_names)

    @classmethod
    @abstractmethod
    def get_specification(cls):
        """Return a specification with AlgorithmType.FETA."""

    @classmethod
    def resolve_inputs(cls, x, y):
        return list(x or []), list(y or [])

    @abstractmethod
    def run(self, x: FedTensor, y: FedTensor):
        """Execute with prepared FedTensor inputs."""
