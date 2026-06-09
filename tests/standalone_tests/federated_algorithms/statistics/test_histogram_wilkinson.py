import math

import numpy as np
import pytest

from exaflow.algorithms.federated.statistics.histogram import WilkinsonHistogram
from exaflow.algorithms.federated.statistics.histogram import _wilkinson_step
from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.utils.aggregators.numpy_aggregator import (
    NumpyAggregator,
)
from tests.standalone_tests.federated_algorithms.utils.federated_algorithm_test import (
    FederatedAlgorithmTest,
)
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    AggregationCoordinator,
)
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    SimulatedAggClient,
)


def _run_single_worker(x: np.ndarray, num_bins: int):
    coordinator = AggregationCoordinator(n_workers=1)
    agg_client = SimulatedAggClient(worker_id=0, coordinator=coordinator)
    hist = WilkinsonHistogram(NumpyAggregator(agg_client))
    return hist.compute(x, num_bins)


def _is_nice_step(step: float) -> bool:
    """Return True iff step == m × 10^k for m in {1, 2, 5}."""
    if step <= 0:
        return False
    k = math.floor(math.log10(step))
    ratio = step / (10**k)
    return any(abs(ratio - m) < 1e-9 for m in [1, 2, 5])


TEST_CASES = [
    {
        "name": "integer_range",
        "x": list(range(0, 101)),
        "bins": 10,
    },
    {
        "name": "small_decimals",
        "x": [0.070 + i * 0.001 for i in range(11)],
        "bins": 10,
    },
    {
        "name": "negative_to_positive",
        "x": list(range(-50, 51)),
        "bins": 20,
    },
    {
        "name": "large_numbers",
        "x": [1000.0 + i * 600.0 for i in range(16)],
        "bins": 15,
    },
    {
        "name": "with_nans",
        "x": [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 8.0, 10.0],
        "bins": 5,
    },
    {
        "name": "bins_snap_to_nice",
        # 7 requested bins over [0,1] → raw_step≈0.143 → snaps to step=0.1 → 10 bins
        "x": [i / 100 for i in range(101)],
        "bins": 7,
    },
    {
        "name": "unbalanced_distribution",
        "x": [1, 1, 1, 1, 1, 1, 1, 2, 5, 10],
        "bins": 5,
    },
    {
        "name": "single_value",
        "x": [3.0, 3.0, 3.0, 3.0],
        "bins": 5,
    },
    {
        "name": "all_nans",
        "x": [np.nan, np.nan, np.nan],
        "bins": 3,
        "expect_error": (BadInputError, "No finite data"),
    },
]


class TestWilkinsonHistogram(FederatedAlgorithmTest):
    """Test suite for WilkinsonHistogram.

    Verifies that the federated result matches centralized computation using
    the same Wilkinson nice-number bin logic.
    """

    def compute_centralized_result(self, X, y, **kwargs):
        x = np.asarray(X, dtype=float)
        num_bins = kwargs["bins"]
        x_clean = x[~np.isnan(x)]
        if len(x_clean) == 0:
            return None
        _, new_min, new_max, new_num_bins = _wilkinson_step(
            float(x_clean.min()), float(x_clean.max()), num_bins
        )
        counts, bin_edges = np.histogram(
            x_clean, bins=new_num_bins, range=(new_min, new_max)
        )
        return counts.astype(float), bin_edges

    def compute_federated_result(self, X, y, *, agg_client, **kwargs):
        aggregator = NumpyAggregator(agg_client)
        hist = WilkinsonHistogram(aggregator)
        return hist.compute(X, kwargs["bins"])

    def compare(self, federated_output, centralized_output, **kwargs):
        fed_counts, fed_edges = federated_output
        cnt_counts, cnt_edges = centralized_output
        assert np.allclose(fed_counts, cnt_counts), (
            f"counts mismatch: federated={fed_counts}, centralized={cnt_counts}"
        )
        assert np.allclose(fed_edges, cnt_edges), (
            f"edges mismatch: federated={fed_edges}, centralized={cnt_edges}"
        )

    @pytest.mark.parametrize("case", TEST_CASES, ids=[c["name"] for c in TEST_CASES])
    def test_federated_algorithm_with_one_worker(self, case):
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
        x = np.array(case["x"], dtype=float)
        if "expect_error" in case:
            error_class, match = case["expect_error"]
            with pytest.raises(error_class, match=match):
                self.run_comparison(
                    X=x, y=np.zeros(len(x)), n_workers=3, bins=case["bins"]
                )
            return
        self.run_comparison(X=x, y=np.zeros(len(x)), n_workers=3, bins=case["bins"])

    def test_actual_bins_may_differ_from_requested(self):
        """Confirm that snapping can change the bin count relative to the hint."""
        # [0, 1] with 7 bins → raw_step ≈ 0.143 → snaps to step=0.1 → 10 bins
        _, _, _, new_num_bins = _wilkinson_step(0.0, 1.0, 7)
        assert new_num_bins != 7

    @pytest.mark.parametrize(
        "data_min, data_max, num_bins, min_step, expected_step",
        [
            (0.0, 7.0, 20, 0.0, 0.2),
            (0.0, 7.0, 20, 1.0, 1.0),
            (0.0, 100.0, 10, 1.0, 10.0),
            (3.0, 3.0, 5, 1.0, 1.0),
        ],
        ids=[
            "no_floor_allows_sub_unit_step",
            "floor_raises_sub_unit_step_to_one",
            "floor_below_natural_step_is_a_noop",
            "degenerate_range_respects_floor",
        ],
    )
    def test_min_step_governs_resulting_step(
        self, data_min, data_max, num_bins, min_step, expected_step
    ):
        """min_step floors the step from below without overriding a larger
        natural step — including for a zero-width (degenerate) range."""
        step, *_ = _wilkinson_step(data_min, data_max, num_bins, min_step=min_step)
        assert step == expected_step

    @pytest.mark.parametrize(
        "data_min, data_max, expect_integer_edges",
        [
            (0.0, 1.0, True),
            (1.0, 5.0, True),
            (0.0, 7.0, True),
            (3.0, 15.0, True),
            (0.0, 1_000_000.0, False),
        ],
        ids=["binary", "likert", "small_range", "gcs_admiss", "wide_range_unaffected"],
    )
    def test_min_step_uses_integer_edges_when_step_collapses_to_one(
        self, data_min, data_max, expect_integer_edges
    ):
        """When min_step=1.0 collapses the step to exactly 1, bin edges must
        be whole numbers with one bin per integer value. Wider steps keep the
        regular nice-number alignment where multiple integers per bin is expected."""
        step, new_min, new_max, new_num_bins = _wilkinson_step(
            data_min, data_max, 20, min_step=1.0
        )
        if expect_integer_edges:
            assert step == 1.0
            assert new_min == math.floor(data_min)
            assert new_max == math.ceil(data_max) + 1
            assert new_num_bins == int(new_max - new_min)
            x = np.arange(data_min, data_max + 1)
            counts, edges = np.histogram(x, bins=new_num_bins, range=(new_min, new_max))
            assert np.allclose(edges, np.arange(new_min, new_max + 1))
            assert counts.tolist() == [1] * len(x), (
                f"expected one count per distinct integer, got {counts.tolist()}"
            )
        else:
            assert step > 1.0
            assert new_min % step == 0
            assert new_max % step == 0

    def test_compute_with_min_step_keeps_integers_distinct(self):
        """WilkinsonHistogram.compute must thread min_step through so that
        each integer value gets counted in its own bin instead of being
        merged with its neighbour."""
        coordinator = AggregationCoordinator(n_workers=1)
        agg_client = SimulatedAggClient(worker_id=0, coordinator=coordinator)
        hist = WilkinsonHistogram(NumpyAggregator(agg_client))
        x = np.array(list(range(0, 8)), dtype=float)

        counts, bin_edges = hist.compute(x, num_bins=20, min_step=1.0)

        assert np.allclose(bin_edges, np.arange(0, 9))
        assert counts.tolist() == [1.0] * len(x)

    def test_invalid_ndim(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        hist = WilkinsonHistogram(NumpyAggregator(None))
        with pytest.raises(BadInputError, match="must be a 1D array"):
            hist.compute(x, num_bins=5)


# --- Property tests: verify output correctness independently of _wilkinson_step ---

PROPERTY_CASES = [c for c in TEST_CASES if "expect_error" not in c]


@pytest.mark.parametrize(
    "case", PROPERTY_CASES, ids=[c["name"] for c in PROPERTY_CASES]
)
def test_step_is_nice_number(case):
    """Step must be m × 10^k where m ∈ {1, 2, 5}."""
    x = np.array(case["x"], dtype=float)
    x_clean = x[~np.isnan(x)]
    step, _, _, _ = _wilkinson_step(
        float(x_clean.min()), float(x_clean.max()), case["bins"]
    )
    assert _is_nice_step(step), f"step={step} is not a nice number"


NON_DEGENERATE_CASES = [c for c in PROPERTY_CASES if len(set(c["x"])) > 1]


@pytest.mark.parametrize(
    "case", NON_DEGENERATE_CASES, ids=[c["name"] for c in NON_DEGENERATE_CASES]
)
def test_bin_edges_are_multiples_of_step(case):
    """new_min and new_max must be exact multiples of the step."""
    x = np.array(case["x"], dtype=float)
    x_clean = x[~np.isnan(x)]
    step, new_min, new_max, _ = _wilkinson_step(
        float(x_clean.min()), float(x_clean.max()), case["bins"]
    )
    assert abs(new_min / step - round(new_min / step)) < 1e-9, (
        f"new_min={new_min} is not a multiple of step={step}"
    )
    assert abs(new_max / step - round(new_max / step)) < 1e-9, (
        f"new_max={new_max} is not a multiple of step={step}"
    )


@pytest.mark.parametrize(
    "case", PROPERTY_CASES, ids=[c["name"] for c in PROPERTY_CASES]
)
def test_data_contained_in_range(case):
    """All non-NaN values must fall within [new_min, new_max]."""
    x = np.array(case["x"], dtype=float)
    x_clean = x[~np.isnan(x)]
    _, new_min, new_max, _ = _wilkinson_step(
        float(x_clean.min()), float(x_clean.max()), case["bins"]
    )
    assert x_clean.min() >= new_min - 1e-9, (
        f"data min {x_clean.min()} < new_min {new_min}"
    )
    assert x_clean.max() <= new_max + 1e-9, (
        f"data max {x_clean.max()} > new_max {new_max}"
    )


@pytest.mark.parametrize(
    "case", PROPERTY_CASES, ids=[c["name"] for c in PROPERTY_CASES]
)
def test_counts_sum_to_non_nan_count(case):
    """Bin counts must sum to the number of non-NaN values."""
    x = np.array(case["x"], dtype=float)
    counts, _ = _run_single_worker(x, case["bins"])
    expected = int(np.sum(~np.isnan(x)))
    assert int(np.sum(counts)) == expected, (
        f"counts sum {int(np.sum(counts))} != non-NaN count {expected}"
    )
