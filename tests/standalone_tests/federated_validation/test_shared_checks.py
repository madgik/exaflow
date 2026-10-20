from numbers import Integral

import numpy as np
import pandas as pd
import pytest

from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.validation import as_scalar
from exaflow.algorithms.federated.validation import check_choice
from exaflow.algorithms.federated.validation import check_dataframe
from exaflow.algorithms.federated.validation import check_dimensions
from exaflow.algorithms.federated.validation import check_feature_names
from exaflow.algorithms.federated.validation import check_interval
from exaflow.algorithms.federated.validation import check_named_steps
from exaflow.algorithms.federated.validation import check_parameter_ranges
from exaflow.algorithms.federated.validation import check_scalar
from exaflow.algorithms.federated.validation import check_survival_inputs
from exaflow.algorithms.federated.validation import check_survival_target


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, "private-value", None, [1]])
def test_scalar_validation_rejects_invalid_values_safely(value):
    with pytest.raises(BadInputError) as error:
        check_scalar(value, name="alpha", minimum=0, maximum=1)
    assert "private-value" not in str(error.value)


@pytest.mark.parametrize("value", [0, 1, 0.5, np.float64(0.5)])
def test_scalar_closed_bounds_preserve_valid_values(value):
    assert check_scalar(value, name="alpha", minimum=0, maximum=1) is value


@pytest.mark.parametrize("value", [0, 1, -0.1, 1.1])
def test_scalar_open_bounds_reject_endpoints_and_out_of_range_values(value):
    with pytest.raises(BadInputError):
        check_scalar(
            value,
            name="quantile fold",
            minimum=0,
            maximum=1,
            include_boundaries="neither",
        )


@pytest.mark.parametrize("value", [1.5, "2", np.nan])
def test_integer_parameter_does_not_silently_coerce(value):
    with pytest.raises(BadInputError):
        check_scalar(value, name="n_splits", minimum=2, target_type=Integral)


@pytest.mark.parametrize("value", [2, np.int64(2), 10**100])
def test_integer_check_accepts_python_and_numpy_integers(value):
    assert (
        check_scalar(value, name="iterations", minimum=1, target_type=Integral) is value
    )


@pytest.mark.parametrize(
    "value, converter, expected", [("2", int, 2), (2.8, int, 2), ("0.5", float, 0.5)]
)
def test_existing_scalar_conversions_are_explicit(value, converter, expected):
    assert as_scalar(value, converter, name="parameter") == expected


@pytest.mark.parametrize("value", [None, "private-value", float("inf")])
def test_scalar_conversion_errors_are_safe(value):
    with pytest.raises(BadInputError) as error:
        as_scalar(value, int, name="iterations")
    assert "private-value" not in str(error.value)


@pytest.mark.parametrize("value", ["private-value", None, [], np.array([1])])
def test_choice_errors_do_not_echo_supplied_values(value):
    with pytest.raises(BadInputError) as error:
        check_choice(value, ("simple", "wilkinson"), name="histogram_type")
    assert "private-value" not in str(error.value)


@pytest.mark.parametrize(
    "bounds", [[0], [0, 1, 2], [1, 0], [0, np.nan], [0, np.inf], [[0, 1]]]
)
def test_interval_checks_shape_order_and_finiteness(bounds):
    with pytest.raises(BadInputError):
        check_interval(bounds)


def test_equal_histogram_bounds_are_explicitly_allowed():
    check_interval((2, 2), strict=False)
    with pytest.raises(BadInputError):
        check_interval((2, 2))


def test_dataframe_validation_preserves_missing_values_and_duplicate_columns():
    data = pd.DataFrame([[np.nan, 2]], columns=["x", "x"])
    before = data.copy()
    check_dataframe(data, ["x"])
    pd.testing.assert_frame_equal(data, before)
    with pytest.raises(BadInputError, match="required columns"):
        check_dataframe(data, ["missing"])
    with pytest.raises(BadInputError, match="DataFrame"):
        check_dataframe([[1]])


@pytest.mark.parametrize(
    "steps",
    [[("x", object()), ("x", object())], [("", object())], [("x",)], [object()]],
)
def test_component_validation_rejects_bad_structure_and_duplicate_names(steps):
    with pytest.raises(BadInputError):
        check_named_steps(steps)


def test_component_validation_retains_empty_pipeline_behavior():
    check_named_steps([])


def test_rank_checks_do_not_convert_arrays_or_impose_sample_minima():
    values = np.empty((0, 2), dtype=object)
    check_dimensions(values, (2,))
    assert values.dtype == object
    with pytest.raises(BadInputError):
        check_dimensions(values, (1,))


def test_feature_names_are_optional_and_need_not_be_unique():
    check_feature_names(None, 2)
    check_feature_names(["x", "x"], 2)
    with pytest.raises(BadInputError):
        check_feature_names(["x"], 2)


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_positive_parameter_ranges_reject_invalid_values(value):
    with pytest.raises(BadInputError):
        check_parameter_ranges({"tol": value}, positive=("tol",))


def test_nonnegative_parameter_ranges_allow_zero():
    check_parameter_ranges({"ridge": 0}, nonnegative=("ridge",))


@pytest.mark.parametrize(
    "target", [[], [[1]], [[1, 0, 2]], [[np.nan, 0]], [[1, 2]], [[1, np.inf]]]
)
def test_survival_target_checks_the_shared_encoded_contract(target):
    with pytest.raises(BadInputError):
        check_survival_target(target)


def test_survival_normalization_and_local_single_class_are_retained():
    times, events = check_survival_target([[1, 0], [2, 0]])
    features, times, events = check_survival_inputs(
        [[1], [2]], times, events, name="Cox"
    )
    assert features.shape == (2, 1)
    np.testing.assert_array_equal(events, [0, 0])


@pytest.mark.parametrize(
    "features, times, events",
    [
        ([], [], []),
        ([[1]], [0], [1]),
        ([[1]], [np.inf], [1]),
        ([[1]], [1, 2], [1]),
        ([[1]], [1], [2]),
        ([[np.nan]], [1], [1]),
    ],
)
def test_survival_local_checks_reject_invalid_inputs(features, times, events):
    with pytest.raises(BadInputError):
        check_survival_inputs(features, times, events, name="Cox")


def test_unbounded_clipping_intervals_and_equal_finite_limits_are_explicit():
    check_interval((-np.inf, np.inf), finite=False, strict=False)
    check_interval((0, 0), finite=False, strict=False)
    with pytest.raises(BadInputError):
        check_interval((np.inf, np.inf), finite=False, strict=False)
    with pytest.raises(BadInputError):
        check_interval((0, np.nan), finite=False, strict=False)
    check_scalar(np.inf, name="upper_bound", minimum=1, finite=False)
    check_feature_names([], 2)
