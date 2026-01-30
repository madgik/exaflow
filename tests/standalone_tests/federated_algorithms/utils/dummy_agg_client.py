import numpy as np


class DummyAggClient:
    """Local aggregator for standalone federated algorithm tests."""

    def sum(self, value):
        return np.asarray(value, dtype=float)

    def min(self, value):
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return arr.reshape(1)
        return arr

    def max(self, value):
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return arr.reshape(1)
        return arr
