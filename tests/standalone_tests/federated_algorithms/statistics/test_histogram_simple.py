import numpy as np
import pytest

from exaflow.algorithms.federated.statistics.histogram_base import SimpleHistogram
from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.utils.aggregators.numpy_aggregator import (
    NumpyAggregator,
)
from tests.standalone_tests.federated_algorithms.utils.federated_algorithm_test import (
    FederatedAlgorithmTest,
)

TEST_CASES = [
    {
        "name": "basic_small",
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "bins": 3,
    },
    {
        "name": "with_nans",
        "x": [1.0, 2.0, np.nan, 4.0, 5.0, np.nan],
        "bins": 3,
    },
    {
        "name": "one_client_empty_bins",
        # We will split this manually in the test or let FederatedAlgorithmTest handle it
        # If we have 3 workers, we can construct the data so one worker has values in only one range
        "x": [1.0, 1.1, 1.2, 5.0, 5.1, 5.2, 10.0, 10.1, 10.2],
        "bins": 3,
    },
    {
        "name": "extreme_nans",
        # One worker will have only NaNs
        "x": [1.0, 2.0, 3.0, np.nan, np.nan, np.nan],
        "bins": 2,
    },
    {
        "name": "unbalanced_distribution",
        "x": [1, 1, 1, 1, 1, 1, 1, 2, 5, 10],
        "bins": 5,
    },
    {
        "name": "all_nans",
        "x": [np.nan, np.nan, np.nan, np.nan],
        "bins": 3,
        "expect_error": (BadInputError, "No finite data"),
    },
    {
        "name": "single_value",
        "x": [1.0, 1.0, 1.0, 1.0],
        "bins": 3,
    },
]


class TestSimpleHistogram(FederatedAlgorithmTest):
    """Test suite for the SimpleHistogram algorithm.

    Verifies correctness against centralized NumPy execution across
    single and multiple worker configurations, including extreme cases.
    """

    def compute_centralized_result(self, X, y, **kwargs):
        """Compute the ground truth histogram using centralized NumPy."""
        x = np.asarray(X)
        num_bins = kwargs["bins"]

        # Remove NaNs for centralized min/max
        x_non_nan = x[~np.isnan(x)]
        if len(x_non_nan) == 0:
            return None  # Should be handled by expecting error

        min_val = np.min(x_non_nan)
        max_val = np.max(x_non_nan)

        if min_val == max_val:
            max_val = min_val + 1.0

        counts, bin_edges = np.histogram(
            x_non_nan, bins=num_bins, range=(min_val, max_val)
        )
        return counts, bin_edges

    def compute_federated_result(self, X, y, *, agg_client, **kwargs):
        """Compute the federated histogram using the SimpleHistogram algorithm."""
        aggregator = NumpyAggregator(agg_client)
        hist_algo = SimpleHistogram(aggregator)
        num_bins = kwargs["bins"]

        return hist_algo.compute(X, num_bins)

    def compare(self, federated_output, centralized_output, **kwargs):
        """Compare federated output against centralized ground truth."""
        fed_counts, fed_edges = federated_output
        cnt_counts, cnt_edges = centralized_output

        assert np.allclose(fed_counts, cnt_counts)
        assert np.allclose(fed_edges, cnt_edges)

    @pytest.mark.parametrize("case", TEST_CASES, ids=[c["name"] for c in TEST_CASES])
    def test_federated_algorithm_with_one_worker(self, case):
        """Verify the algorithm produces correct results with a single worker."""
        x = np.array(case["x"], dtype=float)
        if "expect_error" in case:
            error_class, match = case["expect_error"]
            with pytest.raises(error_class, match=match):
                self.run_comparison(
                    X=x, y=np.zeros(len(x)), n_workers=1, bins=case["bins"]
                )
            return

        self.run_comparison(X=x, y=np.zeros(len(x)), n_workers=1, bins=case["bins"])

    @pytest.mark.parametrize("case", TEST_CASES, ids=[c["name"] for c in TEST_CASES])
    def test_federated_algorithm_with_multiple_workers(self, case):
        """Verify the algorithm produces correct results across multiple workers."""
        x = np.array(case["x"], dtype=float)

        if "expect_error" in case:
            error_class, match = case["expect_error"]
            with pytest.raises(error_class, match=match):
                self.run_comparison(
                    X=x, y=np.zeros(len(x)), n_workers=3, bins=case["bins"]
                )
            return

        self.run_comparison(X=x, y=np.zeros(len(x)), n_workers=3, bins=case["bins"])

    def test_federated_histogram_empty_client(self):
        """Verify behavior when one of the clients has zero observations."""
        x = np.array([1.0, 10.0], dtype=float)
        # With 3 workers and 2 elements, at least one worker will be empty.
        # This should now pass without ValueError.
        self.run_comparison(X=x, y=np.zeros(len(x)), n_workers=3, bins=2)

    def test_federated_histogram_invalid_ndim(self):
        """Verify that passing a 2D array raises a BadInputError."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
        aggregator = NumpyAggregator(None)  # Mocking aggregator for direct call
        hist_algo = SimpleHistogram(aggregator)

        with pytest.raises(BadInputError, match="must be a 1D array"):
            hist_algo.compute(x, num_bins=2)
