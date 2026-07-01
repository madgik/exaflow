import numpy as np
import pandas as pd
import pytest
from scipy.stats import mannwhitneyu

from exaflow.algorithms.federated.statistics.binned_mann_whitney_u_test import (
    FederatedBinnedMannWhitneyUTest,
)
from exaflow.algorithms.federated.utils.aggregators.numpy_aggregator import (
    NumpyAggregator,
)
from tests.standalone_tests.federated_algorithms.utils.federated_algorithm_test import (
    FederatedAlgorithmTest,
)

# Use a fixed seed so tests are deterministic
rng = np.random.default_rng(42)

# Tolerance for p-value comparison: the binned approximation introduces error
# in rank assignment (values in the same bin get the same rank). With 100 bins
# and n >= 100 per group the error is small. We accept atol=0.05 on p-values
# and require that both methods reach the same significance conclusion at α=0.05.
P_VALUE_ATOL = 0.05

TEST_CASES = [
    {
        "name": "clearly_different_normal",
        "group_a": rng.normal(loc=0.0, scale=1.0, size=200).tolist(),
        "group_b": rng.normal(loc=2.0, scale=1.0, size=200).tolist(),
        "alternative": "two-sided",
    },
    {
        "name": "no_difference_normal",
        "group_a": rng.normal(loc=5.0, scale=1.0, size=200).tolist(),
        "group_b": rng.normal(loc=5.0, scale=1.0, size=200).tolist(),
        "alternative": "two-sided",
    },
    {
        "name": "a_less_than_b",
        "group_a": rng.normal(loc=0.0, scale=1.0, size=150).tolist(),
        "group_b": rng.normal(loc=1.5, scale=1.0, size=150).tolist(),
        "alternative": "less",
    },
    {
        "name": "a_greater_than_b",
        "group_a": rng.normal(loc=3.0, scale=1.0, size=150).tolist(),
        "group_b": rng.normal(loc=1.0, scale=1.0, size=150).tolist(),
        "alternative": "greater",
    },
    {
        "name": "unequal_sizes",
        "group_a": rng.normal(loc=0.0, scale=1.0, size=300).tolist(),
        "group_b": rng.normal(loc=1.0, scale=1.0, size=100).tolist(),
        "alternative": "two-sided",
    },
    {
        "name": "skewed_exponential",
        "group_a": rng.exponential(scale=1.0, size=200).tolist(),
        "group_b": rng.exponential(scale=3.0, size=200).tolist(),
        "alternative": "two-sided",
    },
    {
        "name": "integer_values",
        "group_a": rng.integers(1, 11, size=200).tolist(),
        "group_b": rng.integers(5, 16, size=200).tolist(),
        "alternative": "two-sided",
    },
    {
        "name": "negative_range",
        "group_a": rng.normal(loc=-10.0, scale=2.0, size=150).tolist(),
        "group_b": rng.normal(loc=-5.0, scale=2.0, size=150).tolist(),
        "alternative": "two-sided",
    },
    {
        "name": "large_variance_difference",
        "group_a": rng.normal(loc=0.0, scale=0.5, size=200).tolist(),
        "group_b": rng.normal(loc=0.0, scale=5.0, size=200).tolist(),
        "alternative": "two-sided",
    },
]


def _scipy_reference(group_a, group_b, alternative):
    result = mannwhitneyu(
        group_a,
        group_b,
        alternative=alternative,
        method="asymptotic",
        use_continuity=True,
    )
    return {"u_stat": float(result.statistic), "p_value": float(result.pvalue)}


def _compute_single_worker(sample_a, sample_b, **kwargs):
    """Run ``compute`` on a single simulated worker, returning (result, error)."""
    import threading

    from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
        AggregationCoordinator,
    )
    from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
        SimulatedAggClient,
    )

    coordinator = AggregationCoordinator(n_workers=1)
    out = {"result": None, "error": None}

    def run(agg_client):
        try:
            agg = NumpyAggregator(agg_client)
            test = FederatedBinnedMannWhitneyUTest(agg)
            out["result"] = test.compute(sample_a, sample_b, **kwargs)
        except Exception as exc:  # noqa: BLE001
            out["error"] = exc

    t = threading.Thread(
        target=run, args=(SimulatedAggClient(worker_id=0, coordinator=coordinator),)
    )
    t.start()
    t.join()
    return out["result"], out["error"]


class TestFederatedBinnedMannWhitneyUTest(FederatedAlgorithmTest):
    def compute_centralized_result(self, X, y, **kwargs):
        return _scipy_reference(
            kwargs["sample_a"], kwargs["sample_b"], kwargs["alternative"]
        )

    def compute_federated_result(self, X, y, *, agg_client, **kwargs):
        chunk = X
        grouping = chunk["group"]
        values = chunk["value"].to_numpy(dtype=float)
        sample_a = values[grouping.to_numpy() == "A"]
        sample_b = values[grouping.to_numpy() == "B"]

        aggregator = NumpyAggregator(agg_client)
        test = FederatedBinnedMannWhitneyUTest(aggregator)
        return test.compute(
            sample_a,
            sample_b,
            alternative=kwargs["alternative"],
            num_bins=100,
        )

    def compare(self, federated_output, centralized_output, **kwargs):
        fed_p = federated_output["p_value"]
        ref_p = centralized_output["p_value"]

        print(
            f"\n[binned_mwu] alt={kwargs['alternative']}"
            f"  fed_p={fed_p:.4f}  ref_p={ref_p:.4f}"
            f"  fed_u={federated_output['u_stat']:.1f}"
            f"  ref_u={centralized_output['u_stat']:.1f}"
        )

        # p-values should be close (histogram approximation error)
        assert abs(fed_p - ref_p) <= P_VALUE_ATOL, (
            f"p-value differs too much: federated={fed_p:.4f}, scipy={ref_p:.4f}"
        )

        # Both methods must reach the same significance conclusion at α=0.05
        alpha = 0.05
        assert (fed_p < alpha) == (ref_p < alpha), (
            f"Significance conclusion differs: federated p={fed_p:.4f}, scipy p={ref_p:.4f}"
        )

    @pytest.mark.parametrize("case", TEST_CASES, ids=[c["name"] for c in TEST_CASES])
    def test_federated_algorithm_with_one_worker(self, case):
        sample_a = np.asarray(case["group_a"], dtype=float)
        sample_b = np.asarray(case["group_b"], dtype=float)
        data = pd.DataFrame(
            {
                "group": ["A"] * len(sample_a) + ["B"] * len(sample_b),
                "value": np.concatenate([sample_a, sample_b]),
            }
        )
        self.run_comparison(
            X=data,
            y=np.zeros(len(data), dtype=float),
            n_workers=1,
            sample_a=sample_a,
            sample_b=sample_b,
            alternative=case["alternative"],
        )

    @pytest.mark.parametrize("case", TEST_CASES, ids=[c["name"] for c in TEST_CASES])
    def test_federated_algorithm_with_multiple_workers(self, case):
        sample_a = np.asarray(case["group_a"], dtype=float)
        sample_b = np.asarray(case["group_b"], dtype=float)
        data = pd.DataFrame(
            {
                "group": ["A"] * len(sample_a) + ["B"] * len(sample_b),
                "value": np.concatenate([sample_a, sample_b]),
            }
        )
        self.run_comparison(
            X=data,
            y=np.zeros(len(data), dtype=float),
            n_workers=3,
            sample_a=sample_a,
            sample_b=sample_b,
            alternative=case["alternative"],
        )

    def test_one_sided_against_direction_matches_scipy(self):
        """Continuity correction must follow scipy's orientation even when the
        data runs counter to a one-sided alternative (chosen U below mu).

        Distinct, evenly-spaced values with fine bins give exact ranks, so the
        only thing that can differ from scipy here is the continuity logic.
        ``a`` sits just below ``b``, so ``greater`` is the against-direction
        case that the toward-mu correction used to get wrong.
        """
        import threading

        from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
            AggregationCoordinator,
        )
        from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
            SimulatedAggClient,
        )

        a = np.arange(1, 26, dtype=float)
        b = np.arange(1, 26, dtype=float) + 0.5

        def federated(alternative):
            coordinator = AggregationCoordinator(n_workers=1)
            out = [None]

            def run(agg_client):
                agg = NumpyAggregator(agg_client)
                test = FederatedBinnedMannWhitneyUTest(agg)
                out[0] = test.compute(a, b, alternative=alternative, num_bins=500)

            t = threading.Thread(
                target=run,
                args=(SimulatedAggClient(worker_id=0, coordinator=coordinator),),
            )
            t.start()
            t.join()
            return out[0]

        for alternative in ("two-sided", "less", "greater"):
            fed = federated(alternative)
            ref = _scipy_reference(a, b, alternative)
            assert abs(fed["p_value"] - ref["p_value"]) < 1e-3, (
                f"{alternative}: federated={fed['p_value']:.5f}, "
                f"scipy={ref['p_value']:.5f}"
            )

    def test_all_same_values_raises(self):
        """All-constant data makes sigma=0 which should raise BadInputError."""
        import threading

        from exaflow.algorithms.federated.utils import BadInputError
        from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
            AggregationCoordinator,
        )
        from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
            SimulatedAggClient,
        )

        coordinator = AggregationCoordinator(n_workers=1)
        errors = [None]

        def run(agg_client):
            try:
                agg = NumpyAggregator(agg_client)
                test = FederatedBinnedMannWhitneyUTest(agg)
                test.compute(
                    np.array([5.0] * 50, dtype=float),
                    np.array([5.0] * 50, dtype=float),
                )
            except Exception as exc:
                errors[0] = exc

        t = threading.Thread(
            target=run, args=(SimulatedAggClient(worker_id=0, coordinator=coordinator),)
        )
        t.start()
        t.join()

        assert isinstance(errors[0], BadInputError)

    def test_num_bins_below_two_raises(self):
        """num_bins < 2 cannot form a histogram and must raise BadInputError."""
        from exaflow.algorithms.federated.utils import BadInputError

        a = np.arange(1, 21, dtype=float)
        b = np.arange(1, 21, dtype=float) + 0.5
        _, error = _compute_single_worker(a, b, num_bins=1)
        assert isinstance(error, BadInputError)

    def test_non_finite_values_are_dropped(self):
        """+/-inf and NaN are filtered so results match the finite-only data."""
        a = np.arange(1, 51, dtype=float)
        b = np.arange(1, 51, dtype=float) + 2.0

        a_with_non_finite = np.concatenate([a, [np.inf, -np.inf, np.nan]])
        b_with_non_finite = np.concatenate([b, [np.inf, np.nan]])

        clean, err_clean = _compute_single_worker(a, b, num_bins=100)
        dirty, err_dirty = _compute_single_worker(
            a_with_non_finite, b_with_non_finite, num_bins=100
        )

        assert err_clean is None and err_dirty is None
        assert dirty["n1"] == clean["n1"] == 50
        assert dirty["n2"] == clean["n2"] == 50
        assert dirty["u_stat"] == clean["u_stat"]
        assert abs(dirty["p_value"] - clean["p_value"]) < 1e-9
