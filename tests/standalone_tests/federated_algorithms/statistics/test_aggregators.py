import numpy as np
import pytest

from exaflow.algorithms.federated.utils.aggregators.numpy_aggregator import (
    NumpyAggregator,
)
from tests.standalone_tests.federated_algorithms.utils.federated_algorithm_test import (
    FederatedAlgorithmTest,
)


class TestNumpyAggregatorFed(FederatedAlgorithmTest):
    """Test fed_* methods where each worker contributes a single fixed-shape array."""

    def compute_centralized_result(self, X, y, **kwargs):
        method = kwargs["method"]
        if method == "fed_union":
            if np.issubdtype(X.dtype, np.number):
                return np.unique(X[~np.isnan(X)])
            else:
                return np.unique(X[X != np.array(None)])
        elif method == "fed_sum":
            return np.sum(X, axis=0)
        elif method == "fed_avg":
            return np.mean(X, axis=0)
        elif method == "fed_weighted_avg":
            weights = kwargs["weights"]
            global_w = np.sum(weights)
            # ensure X is float or compatible
            return np.sum(X * np.array(weights)[:, None], axis=0) / global_w
        elif method == "fed_min":
            return np.min(X, axis=0)
        elif method == "fed_max":
            return np.max(X, axis=0)
        return None

    def compute_federated_result(self, X, y, *, agg_client, **kwargs):
        agg = NumpyAggregator(agg_client)
        method = kwargs["method"]
        # X is shape (1, ...) since dataset X is split by run_comparison
        # So X_local is just X[0]
        X_local = X[0]

        if method == "fed_union":
            return agg.fed_union(X_local)
        elif method == "fed_sum":
            return agg.fed_sum(X_local)
        elif method == "fed_avg":
            return agg.fed_avg(X_local)
        elif method == "fed_weighted_avg":
            weights = kwargs["weights"]
            weight = weights[agg_client._worker_id]
            return agg.fed_weighted_avg(X_local, weight)
        elif method == "fed_min":
            return agg.fed_min(X_local)
        elif method == "fed_max":
            return agg.fed_max(X_local)

    def compare(self, federated_output, centralized_output, **kwargs):
        method = kwargs["method"]
        if method == "fed_union":
            if isinstance(federated_output, np.ndarray):
                assert np.array_equal(
                    np.sort(federated_output), np.sort(centralized_output)
                )
            else:
                assert sorted(federated_output) == sorted(centralized_output)
        else:
            assert np.allclose(federated_output, centralized_output, atol=1e-6)

    @pytest.mark.parametrize("method", ["fed_sum", "fed_avg", "fed_min", "fed_max"])
    def test_federated_algorithm_with_one_worker(self, method):
        X = np.random.rand(1, 5)
        self.run_comparison(X=X, y=np.zeros(1), n_workers=1, method=method)

    @pytest.mark.parametrize("method", ["fed_sum", "fed_avg", "fed_min", "fed_max"])
    def test_federated_algorithm_with_multiple_workers(self, method):
        X = np.random.rand(3, 5)
        self.run_comparison(X=X, y=np.zeros(3), n_workers=3, method=method)

    @pytest.mark.parametrize("n_workers", [1, 3])
    def test_fed_weighted_avg(self, n_workers):
        X = np.random.rand(n_workers, 5)
        weights = np.random.rand(n_workers) + 0.1
        self.run_comparison(
            X=X,
            y=np.zeros(n_workers),
            n_workers=n_workers,
            method="fed_weighted_avg",
            weights=weights,
        )

    @pytest.mark.parametrize("n_workers", [1, 3])
    def test_fed_union_numeric(self, n_workers):
        X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])[:n_workers]
        self.run_comparison(
            X=X, y=np.zeros(n_workers), n_workers=n_workers, method="fed_union"
        )

    @pytest.mark.parametrize("n_workers", [1, 3])
    def test_fed_union_categorical(self, n_workers):
        X = np.array([["A", "B"], ["B", "C"], ["C", "D"]], dtype=object)[:n_workers]
        self.run_comparison(
            X=X, y=np.zeros(n_workers), n_workers=n_workers, method="fed_union"
        )


class TestNumpyAggregatorGlobal(FederatedAlgorithmTest):
    """Test global_* methods where each worker contributes data observations (split along axis=0)."""

    def compute_centralized_result(self, X, y, **kwargs):
        method = kwargs["method"]
        if method == "global_sum":
            return np.sum(X, axis=0)
        elif method == "global_count":
            return np.sum(~np.isnan(X), axis=0)
        elif method == "global_avg":
            return np.mean(X, axis=0)
        elif method == "global_min":
            return np.min(X, axis=0)
        elif method == "global_max":
            return np.max(X, axis=0)
        return None

    def compute_federated_result(self, X, y, *, agg_client, **kwargs):
        agg = NumpyAggregator(agg_client)
        method = kwargs["method"]

        if method == "global_sum":
            return agg.global_sum(X)
        elif method == "global_count":
            return agg.global_count(X)
        elif method == "global_avg":
            return agg.global_avg(X)
        elif method == "global_min":
            return agg.global_min(X)
        elif method == "global_max":
            return agg.global_max(X)
        return None

    def compare(self, federated_output, centralized_output, **kwargs):
        # Allow equality tests that can cope with NaNs correctly (np.allclose(equal_nan=True))
        assert np.allclose(
            federated_output, centralized_output, atol=1e-6, equal_nan=True
        )

    @pytest.mark.parametrize(
        "method",
        ["global_sum", "global_count", "global_avg", "global_min", "global_max"],
    )
    def test_federated_algorithm_with_one_worker(self, method):
        X = np.random.rand(10, 5)
        self.run_comparison(X=X, y=np.zeros(10), n_workers=1, method=method)

    @pytest.mark.parametrize(
        "method",
        ["global_sum", "global_count", "global_avg", "global_min", "global_max"],
    )
    def test_federated_algorithm_with_multiple_workers(self, method):
        X = np.random.rand(15, 5)
        self.run_comparison(X=X, y=np.zeros(15), n_workers=3, method=method)

    @pytest.mark.parametrize("method", ["global_count"])
    def test_global_with_nans(self, method):
        X = np.random.rand(15, 5)
        # inject some nans
        X[0, 0] = np.nan
        X[1, 1] = np.nan
        X[5, 2] = np.nan
        # test with multiple workers so NaNs span across parts
        self.run_comparison(X=X, y=np.zeros(15), n_workers=3, method=method)
