from unittest.mock import Mock

import numpy as np
import pytest

from exaflow.algorithms.federated.cluster.kmeans import FederatedKMeans
from exaflow.algorithms.federated.decomposition.pca import FederatedPCA
from exaflow.algorithms.federated.linear_model.cox_regression_classical import (
    FederatedClassicalCoxRegression,
)
from exaflow.algorithms.federated.linear_model.cox_regression_classical import (
    FederatedClassicalCoxRegressionResults,
)
from exaflow.algorithms.federated.linear_model.cox_regression_stacked import (
    FederatedStackedCoxRegression,
)
from exaflow.algorithms.federated.linear_model.cox_regression_stacked import (
    FederatedStackedCoxRegressionResults,
)
from exaflow.algorithms.federated.linear_model.ols import FederatedOLS
from exaflow.algorithms.federated.linear_model.ols import FederatedOLSResults
from exaflow.algorithms.federated.mixed_effects.glmm_binary import FederatedGLMMBinary
from exaflow.algorithms.federated.mixed_effects.glmm_binary import (
    FederatedGLMMBinaryResults,
)
from exaflow.algorithms.federated.mixed_effects.glmm_ordinal import FederatedGLMMOrdinal
from exaflow.algorithms.federated.mixed_effects.glmm_ordinal import (
    FederatedGLMMOrdinalResults,
)
from exaflow.algorithms.federated.mixed_effects.lmm import FederatedLMM
from exaflow.algorithms.federated.mixed_effects.lmm import FederatedLMMResults
from exaflow.algorithms.federated.naive_bayes.naive_bayes_categorical import (
    FederatedCategoricalNB,
)
from exaflow.algorithms.federated.naive_bayes.naive_bayes_gaussian import (
    FederatedGaussianNB,
)
from exaflow.algorithms.federated.utils import BadInputError


@pytest.mark.parametrize(
    "factory",
    [
        FederatedOLS,
        FederatedClassicalCoxRegression,
        FederatedStackedCoxRegression,
        FederatedGLMMBinary,
        FederatedLMM,
        lambda: FederatedGLMMOrdinal(K=2),
        lambda: FederatedGaussianNB(x_vars=["x"], labels=[0, 1]),
        lambda: FederatedCategoricalNB(
            y_var="y", x_vars=["x"], categories={"x": ["a", "b"]}
        ),
    ],
)
def test_model_row_mismatch_fails_without_aggregation(factory):
    model = factory()
    client = Mock()
    y = (
        np.array([[1, 0], [2, 1]])
        if isinstance(
            model, (FederatedClassicalCoxRegression, FederatedStackedCoxRegression)
        )
        else np.array([0, 1])
    )
    kwargs = (
        {"center_ids": np.array([0])}
        if isinstance(model, (FederatedLMM, FederatedGLMMBinary, FederatedGLMMOrdinal))
        else {}
    )
    with pytest.raises(BadInputError):
        model.fit(np.ones((1, 1)), y, agg_client=client, **kwargs)
    assert client.mock_calls == []


@pytest.mark.parametrize(
    "factory, parameter, value",
    [
        (FederatedOLS, "fit_intercept", "false"),
        (FederatedClassicalCoxRegression, "tol", np.nan),
        (FederatedClassicalCoxRegression, "max_iter", 2.5),
        (FederatedStackedCoxRegression, "time_grid_strategy", "private-value"),
        (FederatedLMM, "min_iter", 1000),
        (FederatedLMM, "upper_bound", np.nan),
        (FederatedGLMMBinary, "mode_tol", 0),
        (FederatedGLMMBinary, "clip_log_sigma_bounds", (1, 0)),
        (lambda: FederatedGLMMOrdinal(K=2), "K", 2.5),
        (lambda: FederatedGLMMOrdinal(K=2), "max_step_norm", np.inf),
        (lambda: FederatedKMeans(n_clusters=2), "n_init", 1.5),
        (lambda: FederatedKMeans(n_clusters=2), "tol", np.nan),
        (lambda: FederatedGaussianNB(x_vars=["x"], labels=[0, 1]), "var_smoothing", -1),
    ],
)
def test_mutated_parameters_are_checked_before_aggregation(factory, parameter, value):
    model = factory()
    setattr(model, parameter, value)
    client = Mock()
    kwargs = (
        {"center_ids": np.array([0, 0])}
        if isinstance(model, (FederatedLMM, FederatedGLMMBinary, FederatedGLMMOrdinal))
        else {}
    )
    y = (
        np.array([[1, 0], [2, 1]])
        if isinstance(
            model, (FederatedClassicalCoxRegression, FederatedStackedCoxRegression)
        )
        else np.array([0, 1])
    )
    with pytest.raises(BadInputError) as error:
        model.fit(np.ones((2, 1)), y, agg_client=client, **kwargs)
    assert "private-value" not in str(error.value)
    assert client.mock_calls == []


@pytest.mark.parametrize(
    "factory", [FederatedLMM, FederatedGLMMBinary, lambda: FederatedGLMMOrdinal(K=2)]
)
def test_mixed_effects_scalar_features_fail_as_input_errors(factory):
    client = Mock()
    with pytest.raises(BadInputError):
        factory().fit(2, [0], center_ids=np.array([0]), agg_client=client)
    assert client.mock_calls == []


@pytest.mark.parametrize(
    "factory", [FederatedClassicalCoxRegression, FederatedStackedCoxRegression]
)
def test_survival_feature_names_are_checked_before_aggregation(factory):
    client = Mock()
    with pytest.raises(BadInputError, match="feature name"):
        factory().fit(
            [[1], [2]], [[1, 0], [2, 1]], feature_names=["a", "b"], agg_client=client
        )
    assert client.mock_calls == []


@pytest.mark.parametrize(
    "result_type",
    [
        FederatedOLSResults,
        FederatedClassicalCoxRegressionResults,
        FederatedStackedCoxRegressionResults,
        FederatedLMMResults,
        FederatedGLMMBinaryResults,
        FederatedGLMMOrdinalResults,
    ],
)
def test_prediction_checks_effective_feature_count(result_type):
    result = object.__new__(result_type)
    object.__setattr__(result, "params", np.ones(2))
    object.__setattr__(result, "fit_intercept", False)
    with pytest.raises(BadInputError, match="Prediction features must match"):
        result.predict(np.ones((3, 3)))


@pytest.mark.parametrize(
    "features", [1, np.ones((2, 2, 2)), [[np.nan]], [["private-value"]]]
)
def test_pca_validates_before_aggregation(features):
    client = Mock()
    with pytest.raises(BadInputError):
        FederatedPCA(client).fit(features)
    assert client.mock_calls == []


def test_pca_checks_fitted_state_before_transformation():
    with pytest.raises(BadInputError, match="not fitted"):
        FederatedPCA(Mock()).transform([[1]])


def test_pca_checks_existing_global_count_before_dividing():
    client = Mock()
    client.sum.side_effect = [np.array([1.0]), np.array([1.0]), np.array([1.0])]
    with pytest.raises(BadInputError, match="at least 2"):
        FederatedPCA(client).fit([[1.0]])
    assert client.sum.call_count == 3


@pytest.mark.parametrize(
    "factory, kwargs, attribute, expected",
    [
        (FederatedKMeans, {"n_clusters": "2"}, "n_clusters", 2),
        (FederatedClassicalCoxRegression, {"max_iter": 3.9}, "max_iter", 3),
        (FederatedGLMMBinary, {"fit_intercept": 1}, "fit_intercept", True),
        (FederatedGLMMOrdinal, {"K": 2.9}, "K", 2),
    ],
)
def test_constructor_conversions_remain_algorithm_specific(
    factory, kwargs, attribute, expected
):
    assert getattr(factory(**kwargs), attribute) == expected


@pytest.mark.parametrize(
    "factory, parameter",
    [
        (FederatedLMM, "fit_intercept"),
        (FederatedGLMMBinary, "return_history"),
        (lambda: FederatedGLMMOrdinal(K=2), "fit_intercept"),
    ],
)
def test_mutated_boolean_parameters_are_rejected_locally(factory, parameter):
    client = Mock()
    model = factory()
    setattr(model, parameter, "false")
    with pytest.raises(BadInputError, match="boolean"):
        model.fit([[1]], [0], center_ids=np.array([0]), agg_client=client)
    assert client.mock_calls == []


@pytest.mark.parametrize(
    "factory, settings",
    [
        (FederatedGLMMBinary, {"clip_log_sigma_bounds": None}),
        (FederatedGLMMBinary, {"clip_log_sigma_bounds": (-np.inf, np.inf)}),
        (FederatedGLMMBinary, {"clip_log_sigma_bounds": (0, 0)}),
        (
            lambda **kwargs: FederatedGLMMOrdinal(K=2, **kwargs),
            {"clip_log_sigma_bounds": (-np.inf, np.inf)},
        ),
        (FederatedLMM, {"upper_bound": np.inf}),
    ],
)
def test_existing_unbounded_and_fixed_variance_settings_pass_validation(
    factory, settings
):
    client = Mock()
    client.sum.side_effect = RuntimeError("Validation passed")
    with pytest.raises(RuntimeError, match="Validation passed"):
        factory(**settings).fit(
            [[1], [2]], [0, 1], center_ids=np.array([0, 0]), agg_client=client
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: FederatedGaussianNB(x_vars=["x"], labels=["a", "b"]),
        lambda: FederatedCategoricalNB(
            y_var="y", x_vars=["x"], categories={"x": ["a", "b"]}
        ),
    ],
)
def test_ragged_class_labels_are_safe_local_input_errors(factory):
    client = Mock()
    with pytest.raises(BadInputError, match="class labels") as error:
        factory().fit([[0], [1]], [["private-value"], ["a", "b"]], agg_client=client)
    assert "private-value" not in str(error.value)
    assert client.mock_calls == []


def test_ragged_encoded_features_are_safe_local_input_errors():
    client = Mock()
    model = FederatedCategoricalNB(
        y_var="y", x_vars=["x"], categories={"x": ["a", "b"]}
    )
    with pytest.raises(BadInputError, match="rectangular"):
        model.fit([[0], [0, 1]], [0, 1], agg_client=client)
    assert client.mock_calls == []
