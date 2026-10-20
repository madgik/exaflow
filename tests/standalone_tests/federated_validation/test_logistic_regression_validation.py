from unittest.mock import Mock

import numpy as np
import pytest

from exaflow.algorithms.federated.linear_model.logistic_regression import (
    FederatedLogisticRegression,
)
from exaflow.algorithms.federated.linear_model.logistic_regression import (
    FederatedLogisticRegressionResults,
)
from exaflow.algorithms.federated.utils import BadInputError


@pytest.mark.parametrize("value", ["false", 1, None, [True]])
def test_parameter_validation_precedes_estimation(value):
    model = FederatedLogisticRegression(fit_intercept=value)
    model._collect_stats = Mock(side_effect=AssertionError("Estimation must not run"))
    client = Mock()

    with pytest.raises(BadInputError, match="fit_intercept must be a boolean"):
        model.fit([[1.0], [2.0]], [0, 1], agg_client=client)

    model._collect_stats.assert_not_called()
    assert client.mock_calls == []


def test_parameter_validation_checks_current_value():
    model = FederatedLogisticRegression()
    model.fit_intercept = "false"
    with pytest.raises(BadInputError, match="fit_intercept must be a boolean"):
        model._validate_fit_inputs([[1.0]], [0])


@pytest.mark.parametrize("intercept", [True, False, np.bool_(True), np.bool_(False)])
def test_normalization_preserves_flat_feature_conversion(intercept):
    model = FederatedLogisticRegression(fit_intercept=intercept)
    X = np.array([2.0, 4.0, 6.0])
    y = np.array([[0, 1, 0]])
    original_X = X.copy()
    original_y = y.copy()

    features, outcome = model._validate_fit_inputs(X, y)

    expected = X[:, None]
    if intercept:
        expected = np.column_stack([np.ones(3), expected])
    np.testing.assert_array_equal(features, expected)
    np.testing.assert_array_equal(outcome, [[0], [1], [0]])
    np.testing.assert_array_equal(X, original_X)
    np.testing.assert_array_equal(y, original_y)


@pytest.mark.parametrize("intercept", [False, True])
def test_normalization_preserves_transposition_after_intercept(intercept):
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = np.arange(4 if intercept else 3) % 2
    model = FederatedLogisticRegression(fit_intercept=intercept)

    features, outcome = model._validate_fit_inputs(X, y)

    design = np.column_stack([np.ones(2), X]) if intercept else X
    np.testing.assert_array_equal(features, design.T)
    np.testing.assert_array_equal(outcome, y[:, None])


def test_normalization_preserves_scalar_without_intercept():
    model = FederatedLogisticRegression(fit_intercept=False)
    features, outcome = model._validate_fit_inputs(2.0, 1)
    np.testing.assert_array_equal(features, [[2.0]])
    np.testing.assert_array_equal(outcome, [[1.0]])


@pytest.mark.parametrize("intercept", [True, False])
def test_validation_accepts_empty_local_partition(intercept):
    model = FederatedLogisticRegression(fit_intercept=intercept)
    X, y = model._validate_fit_inputs(np.empty((0, 2)), [])
    assert X.shape == (0, 2 + int(intercept))
    assert y.shape == (0, 1)


def test_validation_accepts_intercept_only_features_and_a_single_local_class():
    X, y = FederatedLogisticRegression()._validate_fit_inputs(
        np.empty((3, 0)), [1, 1, 1]
    )
    np.testing.assert_array_equal(X, np.ones((3, 1)))
    np.testing.assert_array_equal(y, np.ones((3, 1)))


@pytest.mark.parametrize(
    "X, y, message",
    [
        ([[1.0], [2.0]], [0, 2], "encoded as 0 or 1"),
        ([[1.0], [2.0]], [0.2, 0.8], "encoded as 0 or 1"),
        ([[1.0], [2.0]], [0, np.nan], "finite values"),
        ([[1.0], [2.0]], [0, np.inf], "finite values"),
        ([[np.nan], [2.0]], [0, 1], "Features"),
        ([[np.inf], [2.0]], [0, 1], "Features"),
        ([["private-value"], [2.0]], [0, 1], "Features"),
        ([[1.0], [2.0]], [0, "private-value"], "Outcome"),
        ([[1.0], [2.0]], [[0], [1, 0]], "Outcome"),
        ([[10**400], [2.0]], [0, 1], "Features"),
        (np.ones((2, 2, 2)), [0, 1], "Features"),
        ([[1.0], [2.0]], [0, 1, 0], "one outcome"),
    ],
)
def test_invalid_inputs_are_rejected_before_estimation(X, y, message):
    model = FederatedLogisticRegression()
    model._collect_stats = Mock(side_effect=AssertionError("Estimation must not run"))
    client = Mock()

    with pytest.raises(BadInputError, match=message) as error:
        model.fit(X, y, agg_client=client)

    assert "private-value" not in str(error.value)
    model._collect_stats.assert_not_called()
    assert client.mock_calls == []


@pytest.mark.parametrize("error_type", [ConnectionError, RuntimeError])
def test_estimation_errors_propagate_after_local_validation(error_type):
    model = FederatedLogisticRegression()
    model._collect_stats = Mock(side_effect=error_type("Estimation failed"))
    client = Mock()

    with pytest.raises(error_type, match="Estimation failed"):
        model.fit([[1.0], [2.0]], [0, 1], agg_client=client)

    assert client.mock_calls == []


@pytest.mark.parametrize(
    "nobs, p, positive, message",
    [(2, 2, 1, "Logistic regression requires"), (10, 2, 2, "Each outcome class")],
)
def test_global_count_validation_preserves_existing_thresholds(
    nobs, p, positive, message
):
    with pytest.raises(BadInputError, match=message):
        FederatedLogisticRegression._handle_logreg_errors(nobs, p, positive)


def test_global_count_validation_accepts_sufficient_counts():
    FederatedLogisticRegression._handle_logreg_errors(10, 2, 3)


@pytest.mark.parametrize("intercept", [True, False])
def test_prediction_validates_effective_feature_count(intercept):
    results = object.__new__(FederatedLogisticRegressionResults)
    results.params = np.array([0.5, 0.25])
    results.fit_intercept = intercept

    with pytest.raises(BadInputError, match="Prediction features must match"):
        results.predict(np.ones((3, 3)))


@pytest.mark.parametrize("value", [np.nan, np.inf, "private-value"])
def test_prediction_validates_numeric_finite_values(value):
    results = object.__new__(FederatedLogisticRegressionResults)
    results.params = np.array([0.5, 0.25])
    results.fit_intercept = True
    with pytest.raises(BadInputError) as error:
        results.predict([[value]])
    assert "private-value" not in str(error.value)


def test_empty_result_prediction_does_not_impose_new_validation(monkeypatch):
    results = object.__new__(FederatedLogisticRegressionResults)
    results.params = np.array([])
    check = Mock(side_effect=AssertionError("Empty results must bypass validation"))
    monkeypatch.setattr(
        "exaflow.algorithms.federated.linear_model.logistic_regression.check_array",
        check,
    )
    results.predict(np.ones((3, 2)))
    check.assert_not_called()
