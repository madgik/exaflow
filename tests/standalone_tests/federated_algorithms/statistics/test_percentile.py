import threading

import numpy as np
import pandas as pd
import pytest

from exaflow.algorithms.federated.statistics.percentile import Percentile
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

rng = np.random.default_rng(42)
rng_int = np.random.default_rng(123)

TEST_CASES = [
    {
        "name": "uniform_q10",
        "x": rng.uniform(0, 100, 200).tolist(),
        "q": 0.10,
    },
    {
        "name": "uniform_q50",
        "x": rng.uniform(0, 100, 200).tolist(),
        "q": 0.50,
    },
    {
        "name": "uniform_q90",
        "x": rng.uniform(0, 100, 200).tolist(),
        "q": 0.90,
    },
    {
        "name": "normal_q25",
        "x": rng.normal(50, 15, 300).tolist(),
        "q": 0.25,
    },
    {
        "name": "normal_q75",
        "x": rng.normal(50, 15, 300).tolist(),
        "q": 0.75,
    },
    {
        "name": "exponential_q50",
        "x": rng.exponential(scale=10, size=250).tolist(),
        "q": 0.50,
    },
    {
        "name": "exponential_q95",
        "x": rng.exponential(scale=10, size=250).tolist(),
        "q": 0.95,
    },
    {
        "name": "skewed_right_q75",
        "x": ([1] * 150 + rng.uniform(2, 50, 50).tolist()),
        "q": 0.75,
    },
    {
        "name": "negative_range_q50",
        "x": rng.uniform(-100, -10, 200).tolist(),
        "q": 0.50,
    },
    {
        "name": "mixed_negative_positive_q25",
        "x": rng.uniform(-50, 50, 300).tolist(),
        "q": 0.25,
    },
    {
        "name": "large_range_q90",
        "x": rng.uniform(0, 10000, 500).tolist(),
        "q": 0.90,
    },
    {
        "name": "small_dataset_q50",
        "x": rng.uniform(0, 10, 20).tolist(),
        "q": 0.50,
    },
    {
        "name": "with_nans_q50",
        "x": rng.uniform(0, 100, 100).tolist() + [float("nan")] * 20,
        "q": 0.50,
    },
    {
        "name": "integers_q33",
        "x": list(range(1, 101)),
        "q": 0.33,
    },
    {
        "name": "bimodal_q25",
        "x": rng.normal(20, 5, 150).tolist() + rng.normal(80, 5, 150).tolist(),
        "q": 0.25,
    },
    {
        "name": "bimodal_q75",
        "x": rng.normal(20, 5, 150).tolist() + rng.normal(80, 5, 150).tolist(),
        "q": 0.75,
    },
]

INTEGER_TEST_CASES = [
    {
        "name": "consecutive_q33",
        "x": list(range(1, 101)),
        "q": 0.33,
    },
    {
        "name": "consecutive_q50",
        "x": list(range(1, 101)),
        "q": 0.50,
    },
    {
        "name": "consecutive_q90",
        "x": list(range(1, 101)),
        "q": 0.90,
    },
    {
        "name": "dice_rolls_q50",
        "x": rng_int.integers(1, 7, size=300).tolist(),
        "q": 0.50,
    },
    {
        "name": "ratings_1_5_q25",
        "x": rng_int.integers(1, 6, size=200).tolist(),
        "q": 0.25,
    },
    {
        "name": "ratings_1_5_q75",
        "x": rng_int.integers(1, 6, size=200).tolist(),
        "q": 0.75,
    },
    {
        "name": "age_like_q50",
        "x": rng_int.integers(18, 91, size=500).tolist(),
        "q": 0.50,
    },
    {
        "name": "negative_integers_q50",
        "x": rng_int.integers(-50, 1, size=200).tolist(),
        "q": 0.50,
    },
    {
        "name": "large_range_q75",
        "x": rng_int.integers(0, 10001, size=1000).tolist(),
        "q": 0.75,
    },
    {
        "name": "skewed_count_q90",
        "x": ([1] * 150 + list(range(2, 51))),
        "q": 0.90,
    },
]


class TestPercentile(FederatedAlgorithmTest):
    def compute_centralized_result(self, X, y, **kwargs):
        x = np.asarray(X, dtype=float)
        x_clean = x[~np.isnan(x)]
        if len(x_clean) == 0:
            return None
        return float(np.percentile(x_clean, kwargs["q"] * 100))

    def compute_federated_result(self, X, y, *, agg_client, **kwargs):
        aggregator = NumpyAggregator(agg_client)
        algo = Percentile(aggregator)
        return algo.compute(
            pd.Series(X),
            kwargs["q"],
            is_integer=kwargs.get("is_integer", False),
        )

    def compare(self, federated_output, centralized_output, **kwargs):
        value, actual_q = federated_output
        x = np.asarray(kwargs.get("x", []), dtype=float)
        x_clean = x[~np.isnan(x)]
        data_range = float(np.max(x_clean) - np.min(x_clean))
        atol = data_range / 10

        print(
            f"\n[percentile] n={len(x_clean)} atol={atol}\n"
            f"  ({value}, {actual_q}) vs ({centralized_output}, {kwargs['q']})"
        )

        assert value is not None
        assert abs(value - centralized_output) <= atol

    @pytest.mark.parametrize("case", TEST_CASES, ids=[c["name"] for c in TEST_CASES])
    def test_federated_algorithm_with_one_worker(self, case):
        x = np.array(case["x"], dtype=float)
        self.run_comparison(X=x, y=np.zeros(len(x)), n_workers=1, q=case["q"], x=x)

    @pytest.mark.parametrize("case", TEST_CASES, ids=[c["name"] for c in TEST_CASES])
    def test_federated_algorithm_with_multiple_workers(self, case):
        x = np.array(case["x"], dtype=float)
        self.run_comparison(X=x, y=np.zeros(len(x)), n_workers=3, q=case["q"], x=x)

    @pytest.mark.parametrize(
        "case", INTEGER_TEST_CASES, ids=[c["name"] for c in INTEGER_TEST_CASES]
    )
    def test_integer_with_one_worker(self, case):
        x = np.array(case["x"], dtype=int)
        self.run_comparison(
            X=x, y=np.zeros(len(x)), n_workers=1, q=case["q"], x=x, is_integer=True
        )

    @pytest.mark.parametrize(
        "case", INTEGER_TEST_CASES, ids=[c["name"] for c in INTEGER_TEST_CASES]
    )
    def test_integer_with_multiple_workers(self, case):
        x = np.array(case["x"], dtype=int)
        self.run_comparison(
            X=x, y=np.zeros(len(x)), n_workers=3, q=case["q"], x=x, is_integer=True
        )

    def test_all_nans_returns_none(self):
        coordinator = AggregationCoordinator(n_workers=1)
        results = [None]
        errors = [None]

        def run(agg_client):
            try:
                aggregator = NumpyAggregator(agg_client)
                algo = Percentile(aggregator)
                results[0] = algo.compute(pd.Series([float("nan")] * 5), 0.5)
            except Exception as exc:
                errors[0] = exc

        t = threading.Thread(
            target=run,
            args=(SimulatedAggClient(worker_id=0, coordinator=coordinator),),
        )
        t.start()
        t.join()

        if errors[0]:
            raise errors[0]

        assert results[0] is None
