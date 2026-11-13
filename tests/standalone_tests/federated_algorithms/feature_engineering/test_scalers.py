"""Numerical regressions retained through the named-column interface."""

import json
from dataclasses import asdict

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

from exaflow.algorithms.federated.feature_engineering.maxabs_scaler import MaxAbsScaler
from exaflow.algorithms.federated.feature_engineering.minmax_scaler import MinMaxScaler
from exaflow.algorithms.federated.feature_engineering.standard_scaler import (
    StandardScaler,
)
from exaflow.algorithms.federated.utils import BadInputError
from tests.standalone_tests.federated_algorithms.feature_engineering.transformation_test_support import (
    fit as _fit,
)
from tests.standalone_tests.federated_algorithms.feature_engineering.transformation_test_support import (
    fit_in_workers as _fit_in_workers,
)
from tests.standalone_tests.federated_algorithms.feature_engineering.transformation_test_support import (
    fit_transform as _fit_transform,
)
from tests.standalone_tests.federated_algorithms.feature_engineering.transformation_test_support import (
    transform as _transform,
)
from tests.standalone_tests.federated_algorithms.utils.dummy_agg_client import (
    DummyAggClient,
)


def test_standard_scaler_uses_two_pass_global_variance():
    parts = [
        np.array([[1000000000000.0], [1000000000000.0 + 1.0]]),
        np.array([[1000000000000.0 + 2.0]]),
    ]
    fitted = _fit_in_workers(StandardScaler, parts)
    pooled = np.vstack(parts)
    reference = _transform(_fit(SklearnStandardScaler(), pooled), pooled)
    actual = np.vstack(
        [_transform(fitted[0], parts[0]), _transform(fitted[1], parts[1])]
    )
    assert np.allclose(actual, reference, rtol=1e-10, atol=1e-08)
    assert json.dumps(asdict(fitted[0].recipe_), allow_nan=False)
    replay = StandardScaler.from_recipe(
        json.loads(json.dumps(asdict(fitted[0].recipe_), allow_nan=False))
    )
    assert np.allclose(_transform(replay, pooled), actual)


def test_standard_scaler_treats_rounding_only_scale_as_constant():
    data = np.array([[0.1], [0.1], [0.1]])
    scaler = _fit(StandardScaler(), data, agg_client=DummyAggClient())
    assert np.allclose(_transform(scaler, data), 0.0)


def test_minmax_scaler_matches_sklearn_and_replays_from_json_recipe():
    parts = [np.array([[-2.0], [0.0]]), np.array([[4.0], [10.0]])]
    fitted = _fit_in_workers(lambda: MinMaxScaler(feature_range=(-1.0, 2.0)), parts)
    pooled = np.vstack(parts)
    reference = _fit(SklearnMinMaxScaler(feature_range=(-1.0, 2.0)), pooled)
    actual = _transform(fitted[0], pooled)
    recipe = json.loads(json.dumps(asdict(fitted[0].recipe_), allow_nan=False))
    replay = MinMaxScaler.from_recipe(recipe)
    assert np.allclose(actual, _transform(reference, pooled))
    assert np.allclose(_transform(replay, pooled), actual)


def test_minmax_and_maxabs_handle_constant_features_deterministically():
    data = np.array([[3.0, -2.0], [3.0, 4.0]])
    minimum = _fit(MinMaxScaler(), data, agg_client=DummyAggClient())
    maximum = _fit(MaxAbsScaler(), data, agg_client=DummyAggClient())
    assert np.allclose(_transform(minimum, data)[:, 0], 0.0)
    assert np.allclose(_transform(maximum, data)[:, 0], 1.0)
    assert np.allclose(_transform(maximum, data)[:, 1], [-0.5, 1.0])


def test_minmax_preserves_missing_values_in_constant_features():
    fitted = _fit(
        MinMaxScaler(), np.array([[3.0], [3.0], [np.nan]]), agg_client=DummyAggClient()
    )
    transformed = _transform(fitted, np.array([[3.0], [np.nan], [9.0]]))
    assert np.array_equal(transformed, [[0.0], [np.nan], [6.0]], equal_nan=True)


def test_maxabs_rejects_feature_without_valid_values():
    with pytest.raises(BadInputError, match="without valid values"):
        _fit(
            MaxAbsScaler(), np.array([[np.nan], [np.nan]]), agg_client=DummyAggClient()
        )


def test_standard_scaler_preserves_real_variation_at_tiny_magnitudes():
    data = np.array([1e-17, 2e-17, 3e-17]).reshape(-1, 1)
    fitted = _fit(StandardScaler(), data, agg_client=DummyAggClient())
    expected = _fit_transform(SklearnStandardScaler(), data)
    np.testing.assert_allclose(_transform(fitted, data), expected, atol=1e-14)


def test_standard_scaler_many_constant_rows_and_sites():
    data = np.tile(np.linspace(0.01, 100, 200), (10000, 1))
    fitted = _fit_in_workers(StandardScaler, np.array_split(data, 10))
    for model in fitted:
        np.testing.assert_array_equal(model.scale_, np.ones(200))
        np.testing.assert_allclose(_transform(model, data[:1]), 0, atol=1e-10)


@pytest.mark.parametrize("feature_range", [(0, 1), (-2, 4)])
def test_minmax_constant_feature_custom_range_matches_sklearn(feature_range):
    train = np.array([[3.0], [3.0], [np.nan]])
    test = np.array([[3.0], [9.0], [np.nan]])
    actual = _fit(MinMaxScaler(feature_range), train, agg_client=DummyAggClient())
    reference = _fit(SklearnMinMaxScaler(feature_range=feature_range), train)
    np.testing.assert_allclose(
        _transform(actual, test), _transform(reference, test), equal_nan=True
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda recipe: recipe.update(scale=[0]),
        lambda recipe: recipe.update(scale=[1, 2]),
        lambda recipe: recipe.update(location=[np.nan]),
    ],
)
def test_scaler_replay_rejects_invalid_fitted_state(mutate):
    model = _fit(StandardScaler(), [[1], [2]], agg_client=DummyAggClient())
    recipe = asdict(model.recipe_)
    mutate(recipe)
    with pytest.raises(BadInputError):
        StandardScaler.from_recipe(recipe)
