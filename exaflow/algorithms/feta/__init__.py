"""Controller-authored FETA tensor algorithms."""

from exaflow.algorithms.feta import federated_numpy
from exaflow.algorithms.feta.tensor import FedTensor

__all__ = ["FedTensor", "federated_numpy"]
