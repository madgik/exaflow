"""Small internal contracts for federated preprocessing primitives."""

from typing import Protocol

import numpy as np

from exaflow.algorithms.federated.utils.agg_client import AggregationClient


class FederatedPreprocessor(Protocol):
    """Fit globally, then transform a local partition deterministically."""

    def fit(self, data: np.ndarray, *, agg_client: AggregationClient): ...

    def transform(self, data: np.ndarray) -> np.ndarray: ...

    def fit_transform(
        self, data: np.ndarray, *, agg_client: AggregationClient
    ) -> np.ndarray: ...
