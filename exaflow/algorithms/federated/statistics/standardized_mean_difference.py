from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from exaflow.algorithms.federated.sql import FederatedSQL
from exaflow.algorithms.federated.utils.aggregators.numpy_aggregator import (
    NumpyAggregator,
)

RESULT_COLUMNS = [
    "group1",
    "group2",
    "n1",
    "n2",
    "mean1",
    "mean2",
    "var1",
    "var2",
    "smd",
]


class FederatedStandardizedMeanDifference:
    """Compute Cohen's d from aggregated sufficient statistics."""

    def __init__(self, agg_client):
        self.aggregator = NumpyAggregator(agg_client)

    def compute(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute Cohen's d between two independent samples."""
        self._validate_sample(x)
        self._validate_sample(y)

        n1 = self.aggregator.global_count(x)
        n2 = self.aggregator.global_count(y)
        if n1 < 2 or n2 < 2:
            raise ValueError("Each group needs at least 2 points for SMD")

        mean1 = self.aggregator.global_avg(x)
        mean2 = self.aggregator.global_avg(y)
        var1 = self._sample_variance(x, n1, mean1)
        var2 = self._sample_variance(y, n2, mean2)
        return self._smd(n1, mean1, var1, n2, mean2, var2)

    def pairwise_by_group(
        self,
        *,
        data: pd.DataFrame,
        value_var: str,
        group_var: str,
        minimum_group_size: int = 2,
    ) -> pd.DataFrame:
        """Compute Cohen's d for every eligible pair of observed groups."""

        def sample_variance(values):
            count = self.aggregator.global_count(values)
            if count < 2:
                return np.nan
            mean = self.aggregator.global_avg(values)
            return self._sample_variance(values, count, mean)

        statistics = (
            FederatedSQL(data, self.aggregator)
            .group_by(group_var)
            .aggregate("count", self.aggregator.global_count, value_var)
            .aggregate("mean", self.aggregator.global_avg, value_var)
            .aggregate("var", sample_variance, value_var)
            .run()
            .dataframe
        )

        results = []
        for group1, group2 in itertools.combinations(statistics.index, 2):
            first = statistics.loc[group1]
            second = statistics.loc[group2]
            n1 = first["count"]
            n2 = second["count"]
            required_count = max(2, minimum_group_size)
            if n1 < required_count or n2 < required_count:
                continue
            results.append(
                {
                    "group1": group1,
                    "group2": group2,
                    "n1": n1,
                    "n2": n2,
                    "mean1": first["mean"],
                    "mean2": second["mean"],
                    "var1": first["var"],
                    "var2": second["var"],
                    "smd": self._smd(
                        n1,
                        first["mean"],
                        first["var"],
                        n2,
                        second["mean"],
                        second["var"],
                    ),
                }
            )
        return pd.DataFrame(results, columns=RESULT_COLUMNS)

    def _sample_variance(self, values, count, mean):
        squared_deviations = self.aggregator.global_sum((values - mean) ** 2)
        return squared_deviations / (count - 1)

    @staticmethod
    def _validate_sample(sample):
        if not isinstance(sample, np.ndarray):
            raise TypeError("Input must be a numpy array")
        if sample.ndim > 1:
            raise ValueError("Input must be a 1D array")

    @staticmethod
    def _smd(n1, mean1, var1, n2, mean2, var2) -> float:
        pooled_variance = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        if pooled_variance <= 0:
            return 0.0
        return float((mean1 - mean2) / np.sqrt(pooled_variance))
