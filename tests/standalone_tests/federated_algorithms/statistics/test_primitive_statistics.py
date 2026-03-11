import numpy as np
import pytest
from scipy import stats

from exaflow.algorithms.federated.statistics.primitive_statistics import (
    PrimitiveStatistics,
)
from exaflow.algorithms.federated.utils.aggregators.numpy_aggregator import (
    NumpyAggregator,
)
from tests.standalone_tests.federated_algorithms.utils.federated_algorithm_test import (
    FederatedAlgorithmTest,
)

# Test cases similar to test_smd.py
TEST_CASES = [
    {
        "name": "basic_small",
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "y": [2.0, 2.5, 3.5, 4.5, 5.5, 6.5],
    },
    {
        "name": "unequal_distribution",
        "x": [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
        "y": [0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5],
    },
    {
        "name": "with_negative_values",
        "x": [-1.0, 0.0, 1.0, 2.0, 3.0, -2.0],
        "y": [5.0, 4.0, 3.0, 2.0, 1.0, 6.0],
    },
]

UNIVARIATE_METHODS = [
    "count",
    "mean",
    "variance",
    "standard_deviation",
    "range",
    "coefficient_of_variation",
    "mean_absolute_deviation",
    "root_mean_square",
    "mean_square",
]

BIVARIATE_METHODS = [
    "covariance",
    "pearson_correlation",
    "standardized_mean_differences",
]

ALL_METHODS = UNIVARIATE_METHODS + BIVARIATE_METHODS


class TestPrimitiveFederatedStats(FederatedAlgorithmTest):
    def compute_centralized_result(self, X, y, **kwargs):
        method = kwargs["method"]
        x = np.asarray(X)
        y = np.asarray(y)

        if method == "count":
            return float(len(x))
        elif method == "mean":
            return np.mean(x)
        elif method == "variance":
            return np.var(x, ddof=kwargs.get("ddof", 0))
        elif method == "standard_deviation":
            return np.std(x, ddof=kwargs.get("ddof", 0))
        elif method == "range":
            return np.ptp(x)
        elif method == "coefficient_of_variation":
            m = np.mean(x)
            if m == 0:
                return 0.0
            return np.std(x) / m
        elif method == "mean_absolute_deviation":
            return np.mean(np.abs(x - np.mean(x)))
        elif method == "root_mean_square":
            return np.sqrt(np.mean(x**2))
        elif method == "mean_square":
            # Implementation in aggregation_stats.py:
            # y = self.aggregator.global_avg(x)
            # z = y - x
            # return self.aggregator.global_avg(z**2)
            # Which is population variance
            return np.var(x)
        elif method == "covariance":
            # Implementation uses global_sum((x-mx)*(y-my)) / (n-ddof)
            ddof = kwargs.get("ddof", 0)
            n = len(x)
            if n <= ddof:
                return 0.0
            return np.cov(x, y, ddof=ddof)[0, 1]
        elif method == "pearson_correlation":
            r, _ = stats.pearsonr(x, y)
            return r
        elif method == "standardized_mean_differences":
            n1, n2 = len(x), len(y)
            mean1, mean2 = np.mean(x), np.mean(y)
            var1, var2 = np.var(x, ddof=1), np.var(y, ddof=1)
            pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            if pooled_sd == 0:
                return 0.0
            return (mean1 - mean2) / pooled_sd
        return None

    def compute_federated_result(self, X, y, *, agg_client, **kwargs):
        aggregator = NumpyAggregator(agg_client)
        stats_algo = PrimitiveStatistics(aggregator)
        method = kwargs["method"]

        if method in UNIVARIATE_METHODS:
            fn = getattr(stats_algo, method)
            if method in ["mean", "variance", "standard_deviation"]:
                return fn(X, ddof=kwargs.get("ddof", 0))
            return fn(X)
        elif method in BIVARIATE_METHODS:
            fn = getattr(stats_algo, method)
            if method == "covariance":
                return fn(X, y, ddof=kwargs.get("ddof", 0))
            return fn(X, y)

        return None

    def compare(self, federated_output, centralized_output, **kwargs):
        if isinstance(federated_output, (list, tuple)):
            for f, c in zip(federated_output, centralized_output):
                assert np.isclose(f, c, atol=1e-7, rtol=1e-7)
        else:
            assert np.isclose(
                federated_output, centralized_output, atol=1e-7, rtol=1e-7
            )

    @pytest.mark.parametrize("case", TEST_CASES, ids=[c["name"] for c in TEST_CASES])
    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_federated_algorithm_with_one_worker(self, case, method):
        x = np.array(case["x"], dtype=float)
        y = np.array(case["y"], dtype=float)
        self.run_comparison(
            X=x,
            y=y,
            n_workers=1,
            method=method,
            ddof=1 if method in ["variance", "standard_deviation", "covariance"] else 0,
        )

    @pytest.mark.parametrize("case", TEST_CASES, ids=[c["name"] for c in TEST_CASES])
    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_federated_algorithm_with_multiple_workers(self, case, method):
        x = np.array(case["x"], dtype=float)
        y = np.array(case["y"], dtype=float)
        self.run_comparison(
            X=x,
            y=y,
            n_workers=3,
            method=method,
            ddof=1 if method in ["variance", "standard_deviation", "covariance"] else 0,
        )
