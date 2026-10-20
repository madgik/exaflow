import numpy as np
import pandas as pd
import pytest

from exaflow.algorithms.federated.feature_engineering.ordinal_encoder import (
    FederatedOrdinalEncoder,
)
from exaflow.algorithms.federated.mixed_effects import mixed_effects_utils
from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.utils import as_numeric_array
from exaflow.algorithms.federated.validation import ArrayPolicy
from exaflow.algorithms.federated.validation import check_array
from exaflow.algorithms.federated.validation import check_category_order


@pytest.mark.parametrize("dtype", [np.int8, np.int64, np.uint64, np.float32])
def test_explicit_array_dtype_preserves_normalized_input(dtype):
    values = np.array([0, 1, 2], dtype=dtype)
    values.setflags(write=False)

    checked = check_array(
        values, policy=ArrayPolicy(dimensions=(1,)), input_name="Outcome", dtype=dtype
    )

    assert checked is values
    assert checked.dtype == dtype
    assert not checked.flags.writeable


def test_default_array_dtype_keeps_existing_float_conversion():
    values = np.array([0, 1, 2], dtype=np.int64)
    checked = check_array(
        values, policy=ArrayPolicy(dimensions=(1,)), input_name="Outcome"
    )
    assert checked.dtype == np.dtype(float)
    np.testing.assert_array_equal(checked, values)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_explicit_dtype_retains_finite_validation(dtype, value):
    with pytest.raises(BadInputError, match="finite"):
        check_array(
            np.array([value], dtype=dtype),
            policy=ArrayPolicy(dimensions=(1,)),
            input_name="Outcome",
            dtype=dtype,
        )


@pytest.mark.parametrize("dtype", [np.int8, np.int64, np.uint64])
def test_mixed_effects_validates_integer_outcomes_without_a_float_copy(
    monkeypatch, dtype
):
    outcome = np.array([0, 1, 2], dtype=dtype)
    validated_outcomes = []

    def record_checked_outcome(data, **kwargs):
        result = check_array(data, **kwargs)
        if kwargs["input_name"] == "Outcome":
            validated_outcomes.append(result)
        return result

    monkeypatch.setattr(mixed_effects_utils, "check_array", record_checked_outcome)
    mixed_effects_utils.validate_inputs(
        np.ones((3, 1)), outcome, np.array(["a", "a", "b"])
    )

    assert len(validated_outcomes) == 1
    assert validated_outcomes[0] is outcome


def test_mixed_effects_keeps_existing_list_outcome_conversion():
    mixed_effects_utils.validate_inputs(
        np.ones((3, 1)), ["0", "1", "2"], np.array(["a", "a", "b"])
    )


@pytest.mark.parametrize("outcome", [[np.nan], [np.inf], [-np.inf]])
def test_mixed_effects_still_rejects_nonfinite_outcomes(outcome):
    with pytest.raises(BadInputError, match="finite"):
        mixed_effects_utils.validate_inputs(
            np.ones((1, 1)), np.asarray(outcome), np.array(["a"])
        )


@pytest.mark.parametrize(
    "series",
    [
        pd.Series([1.0, np.nan]),
        pd.Series([1, 2], dtype=np.int64),
        pd.Series(["1", "2"], dtype=object),
        pd.Series([True, False]),
        pd.Series([1, None], dtype="Int64"),
        pd.Series([1.0, None], dtype="Float64"),
        pd.Series([True, None], dtype="boolean"),
        pd.Series(["1", None], dtype="string"),
        pd.Series([1, None], dtype="category"),
    ],
)
def test_series_conversion_preserves_numpy_values_and_missing_value_behaviour(series):
    try:
        expected = np.asarray(series, dtype=float)
    except (TypeError, ValueError, OverflowError):
        with pytest.raises(BadInputError, match="numeric values"):
            as_numeric_array(series, input_name="Features")
    else:
        actual = as_numeric_array(series, input_name="Features")
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)


def test_numpy_backed_series_conversion_reuses_read_only_values():
    values = np.array([1.0, 2.0])
    values.setflags(write=False)
    series = pd.Series(values, copy=False)
    converted = as_numeric_array(series, input_name="Features")
    assert np.shares_memory(converted, values)
    assert not converted.flags.writeable


@pytest.mark.parametrize("series_dtype", [object, "string"])
def test_series_conversion_errors_do_not_echo_values(series_dtype):
    with pytest.raises(BadInputError) as error:
        as_numeric_array(
            pd.Series(["private-value"], dtype=series_dtype), input_name="Features"
        )
    assert "private-value" not in str(error.value)


def test_category_validation_returns_a_reusable_dtype_in_the_supplied_order():
    dtype = check_category_order(["b", "a"])
    assert isinstance(dtype, pd.CategoricalDtype)
    assert dtype.categories.tolist() == ["b", "a"]
    np.testing.assert_array_equal(
        pd.Categorical(["a", "b", None], dtype=dtype).codes, [1, 0, -1]
    )


@pytest.mark.parametrize("categories", [None, ["a", "a"], ["a", None], [np.nan]])
def test_ordinal_transform_validates_category_mutations_after_fit(categories):
    data = pd.DataFrame({"x": ["a", "b"]})
    encoder = FederatedOrdinalEncoder(categories={"x": ["a", "b"]})
    encoder.fit(agg_client=None, data=data, categorical_vars=["x"])
    encoder.categories_["x"] = categories

    with pytest.raises(BadInputError, match="Category ordering"):
        encoder.transform(data, categorical_vars=["x"])


def test_ordinal_transform_keeps_category_order_and_unknown_sentinels():
    data = pd.DataFrame({"x": ["a", "b", None, "unknown"]})
    encoder = FederatedOrdinalEncoder(categories={"x": ["b", "a"]}, unknown_value=-2)
    encoder.fit(agg_client=None, data=data, categorical_vars=["x"])
    np.testing.assert_array_equal(
        encoder.transform(data, categorical_vars=["x"]).reshape(-1), [1, 0, -2, -2]
    )


def test_ordinal_transform_rejects_unknown_values_when_requested():
    encoder = FederatedOrdinalEncoder(categories={"x": ["a"]}, handle_unknown="error")
    encoder.fit(
        agg_client=None, data=pd.DataFrame({"x": ["a"]}), categorical_vars=["x"]
    )
    with pytest.raises(BadInputError, match="Unknown categories"):
        encoder.transform(pd.DataFrame({"x": ["b"]}), categorical_vars=["x"])


def test_ordinal_transform_keeps_missing_column_sentinels():
    encoder = FederatedOrdinalEncoder(categories={"x": ["a"]})
    encoder.fit(
        agg_client=None, data=pd.DataFrame({"x": ["a"]}), categorical_vars=["x"]
    )
    encoder.categories_["x"] = None
    np.testing.assert_array_equal(
        encoder.transform(pd.DataFrame(index=[0, 1]), categorical_vars=["x"]),
        [[-1], [-1]],
    )
