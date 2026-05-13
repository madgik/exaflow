import numpy as np
import pandas as pd
import pytest

from exaflow.algorithms.federated.statistics.histogram_simple import (
    CategoricalHistogram,
)
from exaflow.algorithms.federated.utils.aggregators.numpy_aggregator import (
    NumpyAggregator,
)
from tests.standalone_tests.federated_algorithms.utils.federated_algorithm_test import (
    FederatedAlgorithmTest,
)

CATEGORICAL_TEST_CASES = [
    {
        "name": "basic_strings",
        "x": ["A", "B", "A", "C", "B", "A"],
    },
    {
        "name": "with_nans",
        "x": ["A", "B", None, "C", np.nan, "A"],
    },
    {
        "name": "one_client_missing_category",
        "x": ["A", "A", "A", "B", "B", "C"],
        "n_workers": 3,
    },
    {
        "name": "all_nans",
        "x": [None, np.nan, None],
    },
    {
        "name": "empty_client_and_missing_category",
        "x": ["A", "B", None, None, "A", "C"],
        "n_workers": 3,
    },
    {
        "name": "local_empty_global_not_empty",
        "x": ["A", "B", "A", None, None],
        "n_workers": 2,
    },
    {
        "name": "integer_categories",
        "x": [1, 2, 1, 3, 2, 1],
    },
    {
        "name": "integer_categories_multi_worker",
        "x": [10, 20, 10, 30, 20, 10, 40, 50, 40],
        "n_workers": 3,
    },
]


class TestCategoricalHistogram(FederatedAlgorithmTest):
    """Test suite for the CategoricalHistogram algorithm."""

    def compute_centralized_result(self, X, y, **kwargs):
        """Compute ground truth frequency of categories using centralized logic."""
        x = np.asarray(X, dtype=object)

        local_counts_dict = {}
        for item in x:
            if not pd.isna(item):
                s_item = str(item)
                local_counts_dict[s_item] = local_counts_dict.get(s_item, 0) + 1

        unique_vals = sorted(local_counts_dict.keys())

        if not unique_vals:
            return np.array([], dtype=object), np.array([], dtype=float)

        counts = [float(local_counts_dict[val]) for val in unique_vals]
        return np.array(unique_vals, dtype=object), np.array(counts, dtype=float)

    def _outputs_equal(self, left, right):
        """Custom comparison to handle string arrays in the result tuple."""
        if isinstance(left, tuple) and isinstance(right, tuple):
            if len(left) != len(right):
                return False
            # First element is categories (strings), second is counts (floats)
            return np.array_equal(left[0], right[0]) and np.allclose(
                left[1], right[1], atol=1e-8, rtol=1e-8
            )
        return super()._outputs_equal(left, right)

    def compute_federated_result(self, X, y, *, agg_client, **kwargs):
        """Compute the federated categorical histogram."""
        aggregator = NumpyAggregator(agg_client)
        hist_algo = CategoricalHistogram(aggregator)
        return hist_algo.compute(X)

    def compare(self, federated_output, centralized_output, **kwargs):
        """Compare federated output against centralized ground truth."""
        fed_cats, fed_counts = federated_output
        cnt_cats, cnt_counts = centralized_output

        assert np.array_equal(fed_cats, cnt_cats)
        assert np.allclose(fed_counts, cnt_counts)

    @pytest.mark.parametrize(
        "case", CATEGORICAL_TEST_CASES, ids=[c["name"] for c in CATEGORICAL_TEST_CASES]
    )
    def test_federated_algorithm_with_one_worker(self, case):
        """Verify the algorithm produces correct results with a single worker."""
        x = np.array(case["x"], dtype=object)
        self.run_comparison(X=x, y=np.zeros(len(x)), n_workers=1)

    @pytest.mark.parametrize(
        "case", CATEGORICAL_TEST_CASES, ids=[c["name"] for c in CATEGORICAL_TEST_CASES]
    )
    def test_federated_algorithm_with_multiple_workers(self, case):
        """Verify the algorithm produces correct results across multiple workers."""
        x = np.array(case["x"], dtype=object)
        n_workers = case.get("n_workers", 3)
        self.run_comparison(X=x, y=np.zeros(len(x)), n_workers=n_workers)
