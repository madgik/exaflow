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
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        p: int = 0,
        agg_client: AggregationClient,
    ) -> dict:
        metrics_per_fold: dict[str, list] = {}

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
