"""Partition, replay and numerical edge-case regressions."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import numpy as np
import pytest

from exaflow.algorithms.federated.feature_engineering.quantile_transformer import (
    QuantileTransformerApprox,
)
from exaflow.algorithms.federated.utils import BadInputError
from tests.standalone_tests.federated_algorithms.feature_engineering.transformation_test_support import (
    fit as _fit,
)
from tests.standalone_tests.federated_algorithms.feature_engineering.transformation_test_support import (
    transform as _transform,
)
from tests.standalone_tests.federated_algorithms.utils.dummy_agg_client import (
    DummyAggClient,
)
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    AggregationCoordinator,
)
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    SimulatedAggClient,
)


@pytest.mark.parametrize("outlier", [-10000.0, 10000.0])
def test_quantiles_remain_accurate_with_one_distant_outlier(outlier):
    values = np.r_[np.random.default_rng(42).normal(size=1999), outlier]
    model = _fit(
        QuantileTransformerApprox(platform_minimum=10),
        values,
        agg_client=DummyAggClient(),
    )
    median = np.interp(0.5, model.references_, model.quantiles_[0])
    assert median == pytest.approx(np.median(values), abs=0.05)
    assert _transform(model, [0.0])[0] == pytest.approx(np.mean(values <= 0), abs=0.02)


def test_quantiles_remain_accurate_for_skewed_lognormal_data():
    values = np.random.default_rng(42).lognormal(mean=0, sigma=2, size=10000)
    model = _fit(
        QuantileTransformerApprox(platform_minimum=10),
        values,
        agg_client=DummyAggClient(),
    )
    references = np.array([0.25, 0.5, 0.75, 0.9])
    estimated = np.interp(references, model.references_, model.quantiles_[0])
    np.testing.assert_allclose(estimated, np.quantile(values, references), rtol=0.05)


def _fit_workers(parts, factories):
    coordinator = AggregationCoordinator(len(parts))

    def fit_one(i):
        try:
            return _fit(
                factories[i](),
                parts[i],
                agg_client=SimulatedAggClient(worker_id=i, coordinator=coordinator),
            )
        except BadInputError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=len(parts)) as executor:
        futures = [executor.submit(fit_one, i) for i in range(len(parts))]
        try:
            return [future.result(timeout=30) for future in futures]
        except Exception as exc:
            coordinator.abort(exc)
            raise


def test_skewed_refinement_is_partition_stable_with_empty_local_subgroups():
    rng = np.random.default_rng(7)
    data = np.column_stack(
        (np.r_[rng.normal(size=1999), 10000], rng.lognormal(0, 2, 2000))
    )
    parts = [data[:1000], data[1000:1999], data[1999:], np.full((2, 2), np.nan)]
    factories = [lambda: QuantileTransformerApprox(platform_minimum=10)] * len(parts)
    models = _fit_workers(parts, factories)
    pooled = _fit(factories[0](), data, agg_client=DummyAggClient())
    for model in models:
        assert asdict(model.recipe_) == asdict(pooled.recipe_)
        np.testing.assert_array_equal(_transform(model, data), _transform(pooled, data))


@pytest.mark.parametrize("n_bins", [2, 16, 256])
def test_final_intervals_enclose_global_ranks_and_respect_minimum_support(n_bins):
    rng = np.random.default_rng(23)
    data = np.r_[rng.normal(-10, 1, 500), rng.normal(10, 2, 500), 10000]
    model = _fit(
        QuantileTransformerApprox(platform_minimum=10, n_quantiles=51, n_bins=n_bins),
        data,
        agg_client=DummyAggClient(),
    )
    diagnostic = model.fit_diagnostics_[0]
    intervals = np.asarray(diagnostic.intervals)
    truth = np.quantile(data, model.references_, method="inverted_cdf")
    assert np.all(truth >= intervals[:, 0])
    assert np.all(truth <= intervals[:, 1])
    assert np.all(np.diff(model.quantiles_[0]) >= 0)
    for (lower, upper), support in zip(
        intervals[1:-1], diagnostic.support_counts[1:-1]
    ):
        assert support >= 10
        assert np.count_nonzero((data >= lower) & (data <= upper)) == support


def test_refinement_preserves_exact_boundary_and_tied_observations():
    data = np.repeat(np.arange(100.0), 20)
    model = _fit(
        QuantileTransformerApprox(platform_minimum=10, n_quantiles=101, n_bins=16),
        data,
        agg_client=DummyAggClient(),
    )
    intervals = np.asarray(model.fit_diagnostics_[0].intervals)
    expected = np.quantile(data, model.references_, method="inverted_cdf")
    assert np.all((intervals[:, 0] <= expected) & (expected <= intervals[:, 1]))
    np.testing.assert_allclose(model.quantiles_[0], expected, atol=1)


@pytest.mark.parametrize("outlier", [-1e100, 1e100])
def test_local_refinement_trims_huge_global_ranges_without_losing_variation(outlier):
    data = np.r_[np.random.default_rng(42).normal(size=1999), outlier]
    model = _fit(
        QuantileTransformerApprox(platform_minimum=10),
        data,
        agg_client=DummyAggClient(),
    )
    median = np.interp(0.5, model.references_, model.quantiles_[0])
    assert median == pytest.approx(np.median(data), abs=0.05)


def test_refinement_and_achieved_accuracy_replay_from_json():
    data = np.random.default_rng(3).lognormal(0, 2, 5000)
    model = _fit(
        QuantileTransformerApprox(platform_minimum=10),
        data,
        agg_client=DummyAggClient(),
    )
    replay = QuantileTransformerApprox.from_recipe(
        json.loads(json.dumps(asdict(model.recipe_), allow_nan=False))
    )
    assert replay.fit_diagnostics_ == model.fit_diagnostics_
    assert replay.max_refinement_rounds == model.max_refinement_rounds
    np.testing.assert_array_equal(_transform(replay, data), _transform(model, data))


def test_minimum_support_stops_refinement_and_records_remaining_uncertainty():
    data = np.arange(10.0)
    model = QuantileTransformerApprox(platform_minimum=10, n_quantiles=3)
    _fit(model, np.tile(data, 2), agg_client=DummyAggClient())
    diagnostic = model.fit_diagnostics_[0]
    assert diagnostic.limited_by_minimum_support
    assert diagnostic.support_counts[1] >= 10
    assert diagnostic.max_interval_width > 0
    assert diagnostic.histogram_rounds <= 3


@pytest.mark.parametrize("rounds", [0, 1, 2])
def test_histogram_round_budget_is_enforced_and_accuracy_is_recorded(rounds):
    data = np.random.default_rng(9).normal(size=10000)
    model = _fit(
        QuantileTransformerApprox(
            platform_minimum=10, n_quantiles=11, max_refinement_rounds=rounds
        ),
        data,
        agg_client=DummyAggClient(),
    )
    diagnostic = model.fit_diagnostics_[0]
    assert diagnostic.histogram_rounds <= rounds + 1
    assert diagnostic.max_interval_width == max(
        (high - low for low, high in diagnostic.intervals)
    )


@pytest.mark.parametrize("n_quantiles", [10, 100])
def test_batched_refinement_cost_does_not_scale_as_round_trips_per_quantile(
    n_quantiles,
):

    class CountingClient(DummyAggClient):
        def __init__(self):
            self.calls = 0
            self.largest_payload = 0

        def record(self, values):
            self.calls += 1
            self.largest_payload = max(self.largest_payload, np.size(values))

        def sum(self, values):
            self.record(values)
            return super().sum(values)

        def min(self, values):
            self.record(values)
            return super().min(values)

        def max(self, values):
            self.record(values)
            return super().max(values)

    data = np.random.default_rng(11).normal(size=(10000, 2))
    client = CountingClient()
    rounds = 2
    model = _fit(
        QuantileTransformerApprox(
            platform_minimum=10, n_quantiles=n_quantiles, max_refinement_rounds=rounds
        ),
        data,
        agg_client=client,
    )
    assert client.calls <= 6 + 3 * data.shape[1] * (rounds + 1)
    assert client.largest_payload <= max(8, (n_quantiles - 2) * model.n_bins)


@pytest.mark.parametrize(
    "mismatch", [{"n_bins": 16}, {"n_quantiles": 20}, {"max_refinement_rounds": 0}]
)
def test_inconsistent_refinement_settings_fail_on_every_site(mismatch):
    factories = [
        lambda: QuantileTransformerApprox(platform_minimum=10),
        lambda: QuantileTransformerApprox(platform_minimum=10, **mismatch),
    ]
    results = _fit_workers([np.arange(2000.0), np.arange(2000.0)], factories)
    assert all(
        (
            isinstance(result, BadInputError) and "settings" in str(result)
            for result in results
        )
    )


@pytest.mark.parametrize("rounds", [-1, True, 1.5, np.nan])
def test_invalid_refinement_budget_is_bad_input(rounds):
    with pytest.raises(BadInputError):
        QuantileTransformerApprox(platform_minimum=10, max_refinement_rounds=rounds)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda recipe: recipe["fit_diagnostics"][0]["support_counts"].__setitem__(1, 1),
        lambda recipe: recipe["fit_diagnostics"][0].update(
            max_interval_width=10000000000.0
        ),
        lambda recipe: recipe["fit_diagnostics"][0].update(histogram_rounds=100),
    ],
)
def test_replay_rejects_invalid_refinement_diagnostics(mutate):
    model = _fit(
        QuantileTransformerApprox(platform_minimum=10, n_quantiles=5),
        np.arange(200.0),
        agg_client=DummyAggClient(),
    )
    recipe = asdict(model.recipe_)
    mutate(recipe)
    with pytest.raises(BadInputError):
        QuantileTransformerApprox.from_recipe(recipe)
