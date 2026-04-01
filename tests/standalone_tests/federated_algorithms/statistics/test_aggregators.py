import numpy as np
import pandas as pd
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
                return np.unique(X[~pd.isna(X)])
        elif method == "fed_sum":
            result = np.nansum(X, axis=0).astype(float)
            result[np.all(np.isnan(X), axis=0)] = np.nan
            return result
        elif method == "fed_avg":
            nan_mask = np.isnan(X)
            filled = np.where(nan_mask, 0.0, X)
            valid_count = np.sum(~nan_mask, axis=0).astype(float)
            return np.where(
                valid_count == 0, np.nan, np.sum(filled, axis=0) / valid_count
            )
        elif method == "fed_weighted_avg":
            weights = kwargs["weights"]
            global_w = np.sum(weights)
            # NaN-safe: exclude NaN positions from both numerator and weight
            result = np.zeros(X.shape[1])
            total_eff_weight = np.zeros(X.shape[1])
            for i, w in enumerate(weights):
                mask = ~np.isnan(X[i])
                result += np.where(mask, X[i], 0.0) * w
                total_eff_weight += np.where(mask, w, 0.0)
            return np.where(total_eff_weight == 0, np.nan, result / total_eff_weight)
        elif method == "fed_min":
            result = np.nanmin(X, axis=0).astype(float)
            result[np.all(np.isnan(X), axis=0)] = np.nan
            return result
        elif method == "fed_max":
            result = np.nanmax(X, axis=0).astype(float)
            result[np.all(np.isnan(X), axis=0)] = np.nan
            return result
        return None

    def compute_federated_result(self, X, y, *, agg_client, **kwargs):
        agg = NumpyAggregator(agg_client)
        method = kwargs["method"]
        # X is shape (1, ...) since dataset X is split by run_comparison
        # So X_local is just X[0]
        X_local = X[0]

        if method == "fed_union":
            _ans = agg.fed_union(X_local)
            return _ans
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
        return None

    def _outputs_equal(self, left, right):
        """Override to handle NaN/None in union results during inter-worker check."""
        if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
            if len(left) != len(right):
                return False
            if np.issubdtype(left.dtype, np.number):
                return np.array_equal(left, right, equal_nan=True)
            else:
                # Object arrays: cast to str so None -> 'None', stable comparison
                return np.array_equal(left.astype(str), right.astype(str))
        return super()._outputs_equal(left, right)

    def compare(self, federated_output, centralized_output, **kwargs):
        method = kwargs["method"]
        if method == "fed_union":
            if isinstance(federated_output, np.ndarray):
                if np.issubdtype(federated_output.dtype, np.number):
                    # Sort NaN-safe: NaNs always go to the end
                    def nan_safe_sort(arr):
                        finite = np.sort(arr[~np.isnan(arr)])
                        nan_count = np.sum(np.isnan(arr))
                        return np.concatenate([finite, [np.nan] * nan_count])

                    assert np.array_equal(
                        nan_safe_sort(federated_output),
                        nan_safe_sort(centralized_output),
                        equal_nan=True,
                    )
                else:
                    assert np.array_equal(
                        np.sort(federated_output.astype(str)),
                        np.sort(centralized_output.astype(str)),
                    )
            else:
                assert sorted(federated_output) == sorted(centralized_output)
        else:
            assert np.allclose(
                federated_output, centralized_output, atol=1e-6, equal_nan=True
            )

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

    @pytest.mark.parametrize("n_workers", [1, 3])
    def test_fed_union_numeric_with_nan(self, n_workers):
        """NaN entries must be excluded from the global union."""
        X = np.array([[1.0, np.nan, 2.0], [2.0, 3.0, np.nan], [np.nan, 3.0, 4.0]])[
            :n_workers
        ]
        self.run_comparison(
            X=X, y=np.zeros(n_workers), n_workers=n_workers, method="fed_union"
        )

    @pytest.mark.parametrize("n_workers", [1, 3])
    def test_fed_union_categorical_with_nan(self, n_workers):
        """None entries in object arrays must be excluded from the global union."""
        X = np.array(
            [["A", None, "B"], ["B", "C", None], [None, "C", "D"]], dtype=object
        )[:n_workers]
        self.run_comparison(
            X=X, y=np.zeros(n_workers), n_workers=n_workers, method="fed_union"
        )

    @pytest.mark.parametrize("n_workers", [1, 3])
    def test_fed_union_categorical_with_nan_and_none(self, n_workers):
        """None and NaN entries in object arrays must be excluded from the global union."""
        X = np.array(
            [["A", None, "B"], ["B", np.nan, None], [np.nan, "C", "D"]], dtype=object
        )[:n_workers]
        self.run_comparison(
            X=X, y=np.zeros(n_workers), n_workers=n_workers, method="fed_union"
        )

    @pytest.mark.parametrize("n_workers", [1, 3])
    def test_fed_sum_with_nan(self, n_workers):
        """NaN values are ignored; a position is NaN only when all workers had NaN."""
        X = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan], [7.0, 8.0, 9.0]])[
            :n_workers
        ]
        self.run_comparison(
            X=X, y=np.zeros(n_workers), n_workers=n_workers, method="fed_sum"
        )

    @pytest.mark.parametrize("n_workers", [1, 3])
    def test_fed_avg_with_nan(self, n_workers):
        """NaN values are ignored; a position is NaN only when all workers had NaN."""
        X = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan], [7.0, 8.0, 9.0]])[
            :n_workers
        ]
        self.run_comparison(
            X=X, y=np.zeros(n_workers), n_workers=n_workers, method="fed_avg"
        )

    @pytest.mark.parametrize("n_workers", [1, 3])
    def test_fed_weighted_avg_with_nan(self, n_workers):
        """NaN positions are excluded from weighted sum and weight denominator."""
        X = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan], [7.0, 8.0, 9.0]])[
            :n_workers
        ]
        weights = np.array([2.0, 1.0, 3.0])[:n_workers]
        self.run_comparison(
            X=X,
            y=np.zeros(n_workers),
            n_workers=n_workers,
            method="fed_weighted_avg",
            weights=weights,
        )

    @pytest.mark.parametrize("n_workers", [1, 3])
    def test_fed_min_with_nan(self, n_workers):
        """NaN values are ignored; a position is NaN only when all workers had NaN."""
        X = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan], [7.0, 8.0, 9.0]])[
            :n_workers
        ]
        self.run_comparison(
            X=X, y=np.zeros(n_workers), n_workers=n_workers, method="fed_min"
        )

    @pytest.mark.parametrize("n_workers", [1, 3])
    def test_fed_max_with_nan(self, n_workers):
        """NaN values are ignored; a position is NaN only when all workers had NaN."""
        X = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan], [7.0, 8.0, 9.0]])[
            :n_workers
        ]
        self.run_comparison(
            X=X, y=np.zeros(n_workers), n_workers=n_workers, method="fed_max"
        )


class TestNumpyAggregatorGlobal(FederatedAlgorithmTest):
    """Test global_* methods where each worker contributes data observations (split along axis=0)."""

    def compute_centralized_result(self, X, y, **kwargs):
        method = kwargs["method"]
        if method == "global_sum":
            result = np.nansum(X, axis=0).astype(float)
            result[np.all(np.isnan(X), axis=0)] = np.nan
            return result
        elif method == "global_count":
            return np.sum(~np.isnan(X), axis=0)
        elif method == "global_avg":
            result = np.nanmean(X, axis=0).astype(float)
            result[np.all(np.isnan(X), axis=0)] = np.nan
            return result
        elif method == "global_min":
            result = np.nanmin(X, axis=0).astype(float)
            result[np.all(np.isnan(X), axis=0)] = np.nan
            return result
        elif method == "global_max":
            result = np.nanmax(X, axis=0).astype(float)
            result[np.all(np.isnan(X), axis=0)] = np.nan
            return result
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

    def _outputs_equal(self, left, right):
        """Override to handle NaN in global results during inter-worker check."""
        if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
            return np.array_equal(left, right, equal_nan=True)
        return super()._outputs_equal(left, right)

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

    @pytest.mark.parametrize(
        "method",
        ["global_sum", "global_count", "global_avg", "global_min", "global_max"],
    )
    def test_global_with_nans(self, method):
        """NaN values are ignored; a column is NaN only when all observations are NaN."""
        X = np.random.rand(15, 5)
        # inject scattered NaNs (span across worker parts)
        X[0, 0] = np.nan
        X[1, 1] = np.nan
        X[5, 2] = np.nan
        # make an entire column all-NaN → result for that column must be NaN
        X[:, 3] = np.nan
        self.run_comparison(X=X, y=np.zeros(15), n_workers=3, method=method)
