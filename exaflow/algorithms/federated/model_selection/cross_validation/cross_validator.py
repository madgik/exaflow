from __future__ import annotations

import numpy as np
import pandas as pd

from exaflow.algorithms.federated.utils.agg_client import AggregationClient
from exaflow.algorithms.federated.utils.interfaces import FederatedEstimator
from exaflow.algorithms.federated.utils.interfaces import FederatedScorer
from exaflow.algorithms.federated.utils.interfaces import FederatedSplitter


class FederatedCrossValidator:
    """Meta-estimator that runs federated cross-validation."""

    def __init__(
        self,
        *,
        estimator: FederatedEstimator,
        splitter: FederatedSplitter,
        scorer: FederatedScorer,
    ) -> None:
        self.estimator = estimator
        self.splitter = splitter
        self.scorer = scorer

    def evaluate(
        self,
        X: np.ndarray | None,
        y: np.ndarray,
        *,
        data: pd.DataFrame | None = None,
        categorical_vars: list[str] | None = None,
        numerical_vars: list[str] | None = None,
        p: int = 0,
        agg_client: AggregationClient,
    ) -> dict:
        metrics_per_fold: dict[str, list] = {}

        if data is not None:
            if categorical_vars is None or numerical_vars is None:
                raise ValueError("categorical_vars and numerical_vars are required.")
            if not hasattr(self.splitter, "split_indices"):
                raise TypeError("Splitter must support split_indices for DataFrame CV.")

            for train_idx, test_idx in self.splitter.split_indices(len(data)):
                train_df = data.iloc[train_idx]
                test_df = data.iloc[test_idx]
                y_train = y[train_idx]
                y_test = y[test_idx]

                results = self.estimator.fit(
                    agg_client=agg_client,
                    data=train_df,
                    y=y_train,
                    categorical_vars=categorical_vars,
                    numerical_vars=numerical_vars,
                )
                X_test = self.estimator.transform(
                    test_df,
                    categorical_vars=categorical_vars,
                    numerical_vars=numerical_vars,
                )
                n_train = int(getattr(results, "nobs", 0))

                local_stats = self.scorer.local(
                    results,
                    X_test,
                    y_test,
                )
                metrics = self.scorer.aggregate(
                    local_stats,
                    agg_client=agg_client,
                    n_train=n_train,
                    p=p,
                )

                metrics_with_obs = dict(metrics)
                metrics_with_obs["n_obs"] = n_train

                for key, value in metrics_with_obs.items():
                    metrics_per_fold.setdefault(key, []).append(value)
        else:
            if X is None:
                raise ValueError("X is required when data is not provided.")
            for X_train, y_train, X_test, y_test in self.splitter.split(X, y):
                results = self.estimator.fit(X_train, y_train, agg_client=agg_client)
                n_train = int(getattr(results, "nobs", 0))

                local_stats = self.scorer.local(
                    results,
                    X_test,
                    y_test,
                )
                metrics = self.scorer.aggregate(
                    local_stats,
                    agg_client=agg_client,
                    n_train=n_train,
                    p=p,
                )

                metrics_with_obs = dict(metrics)
                metrics_with_obs["n_obs"] = n_train

                for key, value in metrics_with_obs.items():
                    metrics_per_fold.setdefault(key, []).append(value)

        return metrics_per_fold
