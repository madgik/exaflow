import numpy as np

from exaflow.algorithms.federated.agg_client import AggregationClient


class DummyAggClient(AggregationClient):
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

    def union(self, values):
        return sorted({val for val in values if val is not None})
