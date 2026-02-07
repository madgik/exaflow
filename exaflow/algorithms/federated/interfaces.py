from __future__ import annotations

from typing import Iterable
from typing import Optional
from typing import Protocol

import numpy as np
import pandas as pd

from exaflow.algorithms.federated.agg_client import AggregationClient


class FederatedEstimatorResults(Protocol):
    """Results-centric interface returned by federated estimators."""

    nobs: int

    def predict(self, X: np.ndarray) -> np.ndarray: ...


class FederatedEstimator(Protocol):
    """Estimator interface that fits on federated data."""

    def fit(
        self, X: np.ndarray, y: np.ndarray, *, agg_client: AggregationClient
    ) -> FederatedEstimatorResults: ...


class FederatedSplitter(Protocol):
    """Data splitter interface (e.g., KFold)."""

    def split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Iterable[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]: ...


class FederatedScorer(Protocol):
    """Scorer interface for federated CV."""

    def local(
        self,
        results: FederatedEstimatorResults,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict: ...

    def aggregate(
        self,
        local_stats: dict,
        *,
        agg_client: AggregationClient,
        n_train: int,
        p: int,
    ) -> dict: ...


class FederatedTransformer(Protocol):
    """Transformer interface for federated feature preparation."""

    def fit(
        self,
        *,
        agg_client: AggregationClient,
        data: pd.DataFrame,
        categorical_vars: list[str],
        numerical_vars: Optional[list[str]] = None,
    ) -> None: ...

    def get_feature_names_out(
        self,
        *,
        categorical_vars: list[str],
        numerical_vars: Optional[list[str]] = None,
    ) -> list[str]: ...

    def transform(
        self,
        data: pd.DataFrame,
        *,
        categorical_vars: list[str],
        numerical_vars: Optional[list[str]] = None,
    ) -> np.ndarray: ...
