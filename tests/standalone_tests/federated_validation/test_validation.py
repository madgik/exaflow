import numpy as np
import pytest

from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.validation import ArrayPolicy
from exaflow.algorithms.federated.validation import ParameterRule
from exaflow.algorithms.federated.validation import check_array
from exaflow.algorithms.federated.validation import check_binary_target
from exaflow.algorithms.federated.validation import check_consistent_length
from exaflow.algorithms.federated.validation import check_feature_schema
from exaflow.algorithms.federated.validation import check_sample_counts
from exaflow.algorithms.federated.validation import validate_parameters


@pytest.mark.parametrize("shape", [(0, 2), (3, 0), (0, 0)])
def test_array_validation_does_not_impose_sample_or_feature_minima(shape):
    array = np.empty(shape)
    assert check_array(array, policy=ArrayPolicy(), input_name="X") is array


def test_array_validation_preserves_writable_input_and_memory_order():
    array = np.asfortranarray([[1.0, 2.0], [3.0, 4.0]])
    array.setflags(write=False)
    checked = check_array(array, policy=ArrayPolicy(), input_name="X")
    assert checked is array
    assert not checked.flags.writeable
    assert checked.flags.f_contiguous


def test_finite_policy_is_explicit():
    values = np.array([[np.nan, np.inf]])
    assert (
        check_array(values, policy=ArrayPolicy(finite=False), input_name="X") is values
    )
    with pytest.raises(BadInputError, match="finite"):
        check_array(values, policy=ArrayPolicy(), input_name="X")


def test_array_error_has_safe_structured_issue():
    with pytest.raises(BadInputError) as error:
        check_array([["private-value"]], policy=ArrayPolicy(), input_name="X")
    assert error.value.issue.code == "invalid_array"
    assert error.value.issue.input_name == "X"
    assert "private-value" not in str(error.value)


@pytest.mark.parametrize("values", [[], [0, 0], [1, 1], [0, 1]])
def test_binary_check_does_not_require_local_classes_or_observations(values):
    check_binary_target(np.asarray(values))


@pytest.mark.parametrize("value", [2.0, -1.0, 0.5, np.nan, np.inf])
def test_binary_check_rejects_invalid_encoded_values(value):
    with pytest.raises(BadInputError, match="encoded as 0 or 1"):
        check_binary_target(np.array([value]))


def test_length_check_rejects_broadcastable_mismatch():
    with pytest.raises(BadInputError, match="one outcome"):
        check_consistent_length(np.ones((3, 2)), np.array([1.0]))


@pytest.mark.parametrize("shape", [(2,), (0, 2), (3, 2)])
def test_feature_count_uses_effective_dimensions(shape):
    check_feature_schema(np.empty(shape), n_features=2)


@pytest.mark.parametrize("value", [True, np.bool_(False)])
def test_parameter_validation_retains_valid_parameter_objects(value):
    parameters = {"enabled": value}
    validate_parameters(
        parameters,
        {"enabled": ParameterRule(types=(bool, np.bool_), description="a boolean")},
    )
    assert parameters["enabled"] is value


def test_invalid_parameter_is_not_normalized():
    with pytest.raises(BadInputError, match="enabled must be a boolean"):
        validate_parameters(
            {"enabled": "false"},
            {"enabled": ParameterRule(types=(bool,), description="a boolean")},
        )


@pytest.mark.parametrize("strict, fails", [(False, False), (True, True)])
def test_count_boundary_is_method_specific(strict, fails):
    if fails:
        with pytest.raises(BadInputError, match="more than 2"):
            check_sample_counts(2, minimum=2, name="Sample", strict=strict)
    else:
        check_sample_counts(2, minimum=2, name="Sample", strict=strict)


def test_programming_errors_are_not_converted_to_input_errors():
    with pytest.raises(KeyError, match="enabled"):
        validate_parameters(
            {}, {"enabled": ParameterRule(types=(bool,), description="a boolean")}
        )
