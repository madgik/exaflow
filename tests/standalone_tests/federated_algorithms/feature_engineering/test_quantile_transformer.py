"""Numerical regressions retained through the named-column interface."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import numpy as np
import pytest

from exaflow.algorithms.federated.feature_engineering.quantile_transformer import (
    QuantileTransformerApprox,
)
from exaflow.algorithms.federated.feature_engineering.quantile_transformer import (
    QuantileTransformerRecipe,
)
from exaflow.algorithms.federated.utils import BadInputError
from tests.standalone_tests.federated_algorithms.feature_engineering.transformation_test_support import (
    fit as _fit,
)
from tests.standalone_tests.federated_algorithms.feature_engineering.transformation_test_support import (
    fit_in_workers as _fit_in_workers,
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


def test_quantile_transformer_is_monotone_and_replays_recipe():
    data = np.arange(40.0).reshape(-1, 1)
    transformer = _fit(
        QuantileTransformerApprox(n_quantiles=5, platform_minimum=10),
        data,
        agg_client=DummyAggClient(),
    )
    transformed = _transform(transformer, data)
    assert np.all(np.diff(transformed[:, 0]) >= 0)
    assert transformed[0, 0] == pytest.approx(0.0)
    assert transformed[-1, 0] == pytest.approx(1.0)
    recipe = json.loads(json.dumps(asdict(transformer.recipe_), allow_nan=False))
    replay = QuantileTransformerApprox.from_recipe(recipe)
    assert np.allclose(_transform(replay, data), transformed)


def test_quantile_transformer_is_partition_stable():
    pooled = np.arange(40.0).reshape(-1, 1)
    fitted = _fit_in_workers(
        lambda: QuantileTransformerApprox(n_quantiles=5, platform_minimum=10),
        [pooled[:3], pooled[3:17], pooled[17:]],
    )
    actual = np.vstack(
        [
            _transform(fitted[0], pooled[:3]),
            _transform(fitted[1], pooled[3:17]),
            _transform(fitted[2], pooled[17:]),
        ]
    )
    assert np.all(np.diff(actual[:, 0]) >= 0)
    assert actual[0, 0] == pytest.approx(0.0)
    assert actual[-1, 0] == pytest.approx(1.0)


def test_quantile_transformer_requires_global_observations_per_interval():
    with pytest.raises(BadInputError, match="global observations"):
        _fit(
            QuantileTransformerApprox(n_quantiles=5, platform_minimum=10),
            np.arange(39.0).reshape(-1, 1),
            agg_client=DummyAggClient(),
        )


def test_quantile_transformer_constant_normal_output_is_centered():
    transformer = _fit(
        QuantileTransformerApprox(
            n_quantiles=5, output_distribution="normal", platform_minimum=10
        ),
        np.ones((40, 1)),
        agg_client=DummyAggClient(),
    )
    assert np.allclose(_transform(transformer, np.ones((2, 1))), 0.0)


@pytest.mark.parametrize(
    "start, stop",
    [(1000000.0, 1000000.0 + 1), (100000000.0, 100000000.0 + 100), (0, 1e-09)],
)
def test_quantile_transformer_does_not_flatten_nonconstant_features(start, stop):
    data = np.linspace(start, stop, 100).reshape(-1, 1)
    fitted = _fit(
        QuantileTransformerApprox(n_quantiles=5, platform_minimum=10),
        data,
        agg_client=DummyAggClient(),
    )
    actual = _transform(fitted, data)[:, 0]
    assert actual[0] == pytest.approx(0)
    assert actual[-1] == pytest.approx(1)
    assert np.all(np.diff(actual) >= 0)
    assert len(np.unique(actual)) > 2


def test_quantile_fit_uses_bounded_aggregation_rounds():

    class CountingClient(DummyAggClient):
        def __init__(self):
            self.calls = 0

        def sum(self, values):
            self.calls += 1
            return super().sum(values)

        def min(self, values):
            self.calls += 1
            return super().min(values)

        def max(self, values):
            self.calls += 1
            return super().max(values)

    data = np.random.default_rng(12).normal(size=(1000, 4))
    client = CountingClient()
    _fit(
        QuantileTransformerApprox(n_quantiles=100, platform_minimum=10),
        data,
        agg_client=client,
    )
    assert client.calls <= 6 + 3 * data.shape[1] * 3


@pytest.mark.parametrize(
    "configured, platform, n, passes",
    [
        (1, 10, 39, False),
        (1, 10, 40, True),
        (20, 10, 79, False),
        (20, 10, 80, True),
        (5, 25, 99, False),
    ],
)
def test_quantile_privacy_uses_stricter_minimum(configured, platform, n, passes):
    transformer = QuantileTransformerApprox(
        n_quantiles=5,
        platform_minimum=platform,
        min_observations_per_interval=configured,
    )
    data = np.arange(float(n)).reshape(-1, 1)
    if passes:
        _fit(transformer, data, agg_client=DummyAggClient())
        assert transformer.recipe_.min_observations_per_interval == max(
            configured, platform
        )
    else:
        with pytest.raises(BadInputError, match="global observations"):
            _fit(transformer, data, agg_client=DummyAggClient())


def test_quantile_privacy_counts_nonmissing_values_per_feature():
    data = np.column_stack((np.arange(50.0), np.arange(50.0)))
    data[:11, 1] = np.nan
    with pytest.raises(BadInputError, match="global observations"):
        _fit_in_workers(
            lambda: QuantileTransformerApprox(n_quantiles=5, platform_minimum=10),
            [data[:25], data[25:]],
        )


def test_quantile_uses_strictest_platform_minimum_across_sites():
    coordinator = AggregationCoordinator(2)

    def fit_one(worker_id):
        model = QuantileTransformerApprox(
            n_quantiles=5, platform_minimum=[10, 20][worker_id]
        )
        _fit(
            model,
            np.arange(40.0).reshape(-1, 1),
            agg_client=SimulatedAggClient(worker_id=worker_id, coordinator=coordinator),
        )
        return model

    with ThreadPoolExecutor(max_workers=2) as executor:
        models = list(executor.map(fit_one, range(2)))
    assert [model.recipe_.min_observations_per_interval for model in models] == [20, 20]
    np.testing.assert_array_equal(models[0].quantiles_, models[1].quantiles_)


def test_quantile_histogram_error_is_bounded_by_bin_width():
    data = np.random.default_rng(5).normal(size=(2000, 1))
    model = _fit(
        QuantileTransformerApprox(n_quantiles=20, platform_minimum=10, n_bins=128),
        data,
        agg_client=DummyAggClient(),
    )
    reference = np.quantile(data[:, 0], model.references_, method="inverted_cdf")
    width = np.ptp(data[:, 0]) / model.n_bins
    np.testing.assert_allclose(model.quantiles_[0], reference, atol=width, rtol=0)


def test_quantile_ties_use_midpoint_references():
    recipe = QuantileTransformerRecipe(
        quantiles=[[0, 0, 0, 1, 1]],
        references=[0, 0.25, 0.5, 0.75, 1],
        output_distribution="uniform",
        min_observations_per_interval=10,
        platform_minimum=10,
        n_bins=256,
        max_refinement_rounds=2,
        fit_diagnostics=[
            {
                "histogram_rounds": 0,
                "intervals": [[0, 0], [0, 0], [0, 0], [1, 1], [1, 1]],
                "support_counts": [0, 10, 10, 10, 0],
                "max_interval_width": 0,
                "limited_by_minimum_support": False,
                "limited_by_refinement_budget": False,
            }
        ],
        feature_names=["f0"],
        categorical_features=[],
    )
    model = QuantileTransformerApprox.from_recipe(recipe)
    np.testing.assert_array_equal(
        _transform(model, [0, 1, np.nan]), [0.25, 0.875, np.nan]
    )


def test_quantile_histogram_handles_adjacent_representable_values():
    low = 1e16
    high = np.nextafter(low, np.inf)
    data = np.repeat([low, high], 20)
    model = _fit(
        QuantileTransformerApprox(n_quantiles=5, platform_minimum=10),
        data,
        agg_client=DummyAggClient(),
    )
    assert np.ptp(_transform(model, [low, high])) > 0
