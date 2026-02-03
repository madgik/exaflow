from __future__ import annotations

import numpy as np

from exaflow.algorithms.federated.agg_client import AggregationClient
from exaflow.algorithms.federated.interfaces import FederatedEstimator
from exaflow.algorithms.federated.interfaces import FederatedScorer
from exaflow.algorithms.federated.interfaces import FederatedSplitter


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
        self, X: np.ndarray, y: np.ndarray, *, p: int, agg_client: AggregationClient
    ) -> dict:
        n_obs_per_fold = []
        rmse_per_fold = []
        r2_per_fold = []
        mae_per_fold = []
        fstat_per_fold = []

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

            n_obs_per_fold.append(n_train)
            rmse_per_fold.append(float(metrics.get("rmse", 0.0)))
            r2_per_fold.append(float(metrics.get("r2", 0.0)))
            mae_per_fold.append(float(metrics.get("mae", 0.0)))
            fstat_per_fold.append(float(metrics.get("f_stat", 0.0)))

        return {
            "n_obs": n_obs_per_fold,
            "rmse": rmse_per_fold,
            "r2": r2_per_fold,
            "mae": mae_per_fold,
            "f_stat": fstat_per_fold,
        }
