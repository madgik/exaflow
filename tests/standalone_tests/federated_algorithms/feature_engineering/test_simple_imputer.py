"""Numerical regressions retained through the named-column interface."""

import json
from dataclasses import asdict

import numpy as np
import pytest

from exaflow.algorithms.federated.feature_engineering.simple_imputer import (
    SimpleImputer,
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


def test_simple_imputer_mean_is_global_and_does_not_mutate_inputs():
    left = np.array([[1.0], [np.nan]])
    right = np.array([[5.0], [9.0]])
    original_left = left.copy()
    transformers = _fit_in_workers(lambda: SimpleImputer("mean"), [left, right])
    assert np.allclose(transformers[0].statistics_, [5.0])
    assert np.allclose(_transform(transformers[1], right), right)
    assert np.array_equal(left, original_left, equal_nan=True)
    assert np.allclose(_transform(transformers[0], left), [[1.0], [5.0]])


def test_simple_imputer_median_constant_and_safe_categorical_mode():
    data = np.array([[1.0], [2.0], [100.0], [np.nan]])
    median = _fit(SimpleImputer("median"), data, agg_client=DummyAggClient())
    constant = _fit(
        SimpleImputer("constant", fill_value=-1), data, agg_client=DummyAggClient()
    )
    categorical = _fit(
        SimpleImputer("most_frequent", categories=[["a", "b"]]),
        np.array([["a"], ["b"], ["a"], [None]], dtype=object),
        agg_client=DummyAggClient(),
    )
    assert median.statistics_[0] == pytest.approx(2.0)
    assert np.array_equal(_transform(constant, data), [[1.0], [2.0], [100.0], [-1.0]])
    assert _transform(categorical, np.array([[None]], dtype=object)).tolist() == [["a"]]


def test_simple_imputer_constant_defaults_to_zero_for_numeric_input():
    data = np.array([[1.0], [np.nan]])
    imputer = _fit(SimpleImputer("constant"), data, agg_client=DummyAggClient())
    assert np.array_equal(_transform(imputer, data), [[1.0], [0.0]])


def test_simple_imputer_rejects_numerical_mode_without_safe_protocol():
    with pytest.raises(BadInputError, match="metadata vocabulary"):
        _fit(
            SimpleImputer("most_frequent"),
            np.array([[1.0], [2.0], [np.nan]]),
            agg_client=DummyAggClient(),
        )


def test_simple_imputer_rejects_numeric_values_stored_as_object():
    with pytest.raises(BadInputError, match="metadata vocabulary"):
        _fit(
            SimpleImputer("most_frequent"),
            np.array([[1.37], [8.21], [1.37]], dtype=object),
            agg_client=DummyAggClient(),
        )


def test_simple_imputer_mode_rejects_globally_all_missing_feature():
    with pytest.raises(BadInputError, match="no valid observations"):
        _fit(
            SimpleImputer("most_frequent", categories=[["a", "b"]]),
            np.array([[None], [None]], dtype=object),
            agg_client=DummyAggClient(),
        )


def test_constant_imputer_numeric_output_supports_numpy_and_json_replay():
    data = np.array([[1.0], [np.nan]])
    fitted = _fit(SimpleImputer("constant"), data, agg_client=DummyAggClient())
    for model in (fitted, SimpleImputer.from_recipe(asdict(fitted.recipe_))):
        actual = _transform(model, data)
        assert actual.dtype.kind == "f"
        assert not np.isnan(actual).any()
        np.testing.assert_array_equal(actual, [[1], [0]])
    assert fitted.fill_value is None


def test_categorical_imputer_json_roundtrip():
    data = np.array([["a"], ["a"], ["b"], [None]], dtype=object)
    model = _fit(
        SimpleImputer("most_frequent", categories=[["a", "b"]]),
        data,
        agg_client=DummyAggClient(),
    )
    replay = SimpleImputer.from_recipe(
        json.loads(json.dumps(asdict(model.recipe_), allow_nan=False))
    )
    assert _transform(replay, data).tolist() == [["a"], ["a"], ["b"], ["a"]]


def test_mode_uses_metadata_vocabulary_without_observed_value_union():

    class CountsOnlyClient(DummyAggClient):
        def union(self, values):
            pytest.fail("Observed categories must not be shared.")

    model = _fit(
        SimpleImputer("most_frequent", categories=[["a", "b"]]),
        [["a"], ["a"], [None]],
        agg_client=CountsOnlyClient(),
    )
    assert _transform(model, [[None]]).tolist() == [["a"]]
