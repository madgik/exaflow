import numpy as np
import pytest

from exaflow.algorithms.federated.statistics.statistical_library import (
    FederatedStatistics,
)
from exaflow.algorithms.federated.utils.aggregators import NumpyAggregator
from tests.standalone_tests.federated_algorithms.utils.federated_algorithm_test import (
    FederatedAlgorithmTest,
)

TEST_CASES = [
    {
        "name": "balanced_small",
        "x": [1.0, 2.0, 3.0, 4.0],
        "y": [2.0, 2.5, 3.5, 4.5],
    },
    {
        "name": "unequal_sizes",
        "x": [1.2, 1.4, 1.6, 1.8, 2.0, 2.2],
        "y": [0.9, 1.1, 1.3],
    },
    {
        "name": "large_diff",
        "x": [10.0, 11.0, 12.0],
        "y": [0.0, 1.0, 2.0],
    },
    {
        "name": "zero_diff",
        "x": [1.0, 2.0, 3.0],
        "y": [1.0, 2.0, 3.0],
    },
]


class TestFederatedSMD(FederatedAlgorithmTest):
    def compute_centralized_result(self, X, y, **kwargs):
        x = kwargs["x_full"]
        y = kwargs["y_full"]

        n1, n2 = len(x), len(y)
        mean1, mean2 = np.mean(x), np.mean(y)
        var1, var2 = np.var(x, ddof=1), np.var(y, ddof=1)
        pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        if pooled_sd == 0:
            return 0.0
        return (mean1 - mean2) / pooled_sd

    def compute_federated_result(self, X, y, *, agg_client, **kwargs):
        aggregator = NumpyAggregator(agg_client)
        stats = FederatedStatistics(aggregator)
        return stats.standardized_mean_differences(X, y)

    def compare(self, federated_output, centralized_output, **kwargs):
        np.testing.assert_allclose(
            federated_output, centralized_output, rtol=1e-7, atol=1e-10
        )

    @pytest.mark.parametrize("case", TEST_CASES, ids=[c["name"] for c in TEST_CASES])
    def test_federated_algorithm_with_one_worker(self, case):
        x = np.array(case["x"], dtype=float)
        y = np.array(case["y"], dtype=float)
        self.run_comparison(X=x, y=y, n_workers=1, x_full=x, y_full=y)

    @pytest.mark.parametrize("case", TEST_CASES, ids=[c["name"] for c in TEST_CASES])
    def test_federated_algorithm_with_multiple_workers(self, case):
        x = np.array(case["x"], dtype=float)
        y = np.array(case["y"], dtype=float)
        self.run_comparison(X=x, y=y, n_workers=3, x_full=x, y_full=y)
