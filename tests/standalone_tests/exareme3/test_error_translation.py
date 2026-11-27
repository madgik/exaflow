from string import Formatter

import numpy as np
import pandas as pd
import pytest

from exaflow.algorithms.exareme3.utils.error_translation import _TRANSLATIONS
from exaflow.algorithms.exareme3.utils.error_translation import KEEP_MESSAGE
from exaflow.algorithms.exareme3.utils.error_translation import (
    translate_federated_error,
)
from exaflow.algorithms.federated.decomposition.pca import FederatedPCA
from exaflow.algorithms.federated.exceptions import FedError
from exaflow.algorithms.federated.exceptions import FedErrorReason
from exaflow.algorithms.federated.exceptions import FedObservationError
from exaflow.algorithms.federated.exceptions import FedStateError
from exaflow.algorithms.federated.exceptions import FedTypeError
from exaflow.algorithms.federated.exceptions import FedValueError
from exaflow.algorithms.federated.linear_model.logistic_regression import (
    FederatedLogisticRegression,
)
from exaflow.algorithms.federated.model_selection.cross_validation import (
    FederatedKFoldSplitter,
)
from exaflow.algorithms.federated.statistics.anova_oneway import FederatedAnovaOneWay
from exaflow.algorithms.federated.statistics.binned_mann_whitney_u_test import (
    FederatedBinnedMannWhitneyUTest,
)
from exaflow.algorithms.federated.statistics.cross_tabular_table import CrossTabTable
from exaflow.algorithms.federated.statistics.fisher_exact import FisherExact
from exaflow.algorithms.federated.utils.aggregators.numpy_aggregator import (
    NumpyAggregator,
)
from exaflow.worker_communication import AnalysisExecutionError
from exaflow.worker_communication import BadUserInput
from exaflow.worker_communication import InsufficientDataError
from tests.standalone_tests.federated_algorithms.utils.dummy_agg_client import (
    DummyAggClient,
)


@pytest.mark.parametrize(
    "exception_type, python_type",
    [
        (FedValueError, ValueError),
        (FedTypeError, TypeError),
        (FedStateError, RuntimeError),
        (FedObservationError, ValueError),
    ],
)
def test_exception_family_preserves_python_semantics(exception_type, python_type):
    error = exception_type(FedErrorReason.MODEL_NOT_FITTED)
    assert isinstance(error, FedError)
    assert isinstance(error, python_type)


@pytest.mark.parametrize("exception_type", [ValueError, RuntimeError, SystemError])
def test_native_policy_targets_can_keep_messages(monkeypatch, exception_type):
    error = FedStateError(FedErrorReason.MODEL_NOT_FITTED)
    monkeypatch.setitem(_TRANSLATIONS, error.reason, (exception_type, KEEP_MESSAGE))
    translated = translate_federated_error(error)
    assert type(translated) is exception_type
    assert str(translated) == str(error)


def test_reason_determines_application_exception():
    # The source class does not make a failure user-facing.
    internal = translate_federated_error(FedValueError(FedErrorReason.MODEL_NOT_FITTED))
    user_input = translate_federated_error(
        FedTypeError(FedErrorReason.TIMES_POSITIVE_REQUIRED)
    )
    observation = translate_federated_error(
        FedObservationError(
            FedErrorReason.OUTLIER_OBSERVATIONS_INSUFFICIENT,
            variable="v",
            num_dtps=2,
            min_row_count=3,
        )
    )
    assert type(internal) is RuntimeError
    assert type(user_input) is BadUserInput
    assert type(observation) is InsufficientDataError


@pytest.mark.parametrize(
    "operation, reason, message",
    [
        pytest.param(
            lambda: FederatedKFoldSplitter(n_splits=3).split_indices(2),
            FedErrorReason.INSUFFICIENT_FOLD_SAMPLES,
            "There are too few usable observations for the requested number of folds. "
            "Reduce the number of folds or adjust the filters.",
            id="fold_samples",
        ),
        pytest.param(
            lambda: FederatedAnovaOneWay(DummyAggClient()).fit(
                groups=[np.arange(3)], categories=["A"]
            ),
            FedErrorReason.INSUFFICIENT_ANOVA_LEVELS,
            "One-way ANOVA requires at least two observed categories in the grouping "
            "variable. Choose a different grouping variable or adjust the filters.",
            id="anova_levels",
        ),
        pytest.param(
            lambda: FederatedAnovaOneWay(DummyAggClient()).fit(
                groups=[np.arange(3), np.array([])], categories=["A", "B"]
            ),
            FedErrorReason.INSUFFICIENT_ANOVA_LEVELS,
            "One-way ANOVA requires at least two observed categories in the grouping "
            "variable. Choose a different grouping variable or adjust the filters.",
            id="anova_levels_after_aggregation",
        ),
        pytest.param(
            lambda: FederatedAnovaOneWay(DummyAggClient()).fit(
                groups=[np.array([1]), np.array([2])], categories=["A", "B"]
            ),
            FedErrorReason.INSUFFICIENT_OBSERVATIONS,
            "There are too few usable observations for this analysis. "
            "Adjust the filters or select additional data.",
            id="anova_samples",
        ),
        pytest.param(
            lambda: FederatedPCA(DummyAggClient()).fit(np.array([[1.0, 2.0]])),
            FedErrorReason.INSUFFICIENT_OBSERVATIONS,
            "There are too few usable observations for this analysis. "
            "Adjust the filters or select additional data.",
            id="pca_samples",
        ),
        pytest.param(
            lambda: CrossTabTable(NumpyAggregator(DummyAggClient())).compute(
                pd.DataFrame({"factor": [None], "outcome": [None]}),
                factor="factor",
                outcome="outcome",
                dropna=True,
            ),
            FedErrorReason.EMPTY_CONTINGENCY_TABLE,
            "There are no usable category pairs for the selected variables. "
            "Check the filters and missing-value settings.",
            id="empty_contingency_table",
        ),
        pytest.param(
            lambda: FederatedBinnedMannWhitneyUTest(
                NumpyAggregator(DummyAggClient())
            ).compute(np.ones(3), np.ones(3)),
            FedErrorReason.ZERO_RANK_VARIATION,
            "The Mann–Whitney U test could not estimate enough variation in the ranks "
            "to calculate a p-value. Check that the selected values vary.",
            id="rank_variation",
        ),
        pytest.param(
            lambda: FisherExact(NumpyAggregator(DummyAggClient())).compute(
                pd.DataFrame({"factor": ["A", "B", "C"], "outcome": ["X", "Y", "X"]}),
                factor="factor",
                outcome="outcome",
                dropna=True,
            ),
            FedErrorReason.FISHER_TABLE_SHAPE,
            "Fisher's Exact Test requires exactly two observed categories in each "
            "variable. Choose two variables that meet this requirement in the "
            "selected data.",
            id="fisher_table_shape",
        ),
    ],
)
def test_statistical_failures_translate_to_ui_messages(operation, reason, message):
    with pytest.raises(FedError) as exc_info:
        operation()

    assert exc_info.value.reason == reason
    assert translate_federated_error(exc_info.value).message == message


def test_other_input_messages_pass_through_verbatim():
    error = FedValueError(FedErrorReason.TIMES_POSITIVE_REQUIRED)

    assert _TRANSLATIONS[error.reason][1] is KEEP_MESSAGE
    assert translate_federated_error(error).message == str(error)


@pytest.mark.parametrize(
    "reason, message",
    [
        (
            FedErrorReason.LOGISTIC_CONVERGENCE_FAILED,
            "Logistic regression could not converge to a stable solution.",
        ),
        (
            FedErrorReason.KMEANS_CENTERS_FINITE_VALUES_REQUIRED,
            "K-means could not compute finite cluster centers.",
        ),
    ],
)
def test_computation_failures_are_visible_without_blaming_input(reason, message):
    translated = translate_federated_error(FedValueError(reason))

    assert type(translated) is AnalysisExecutionError
    assert not isinstance(translated, BadUserInput)
    assert translated.message == message


@pytest.mark.parametrize(
    "reason, target",
    [
        (FedErrorReason.ARRAY_DIMENSIONS_INVALID, ValueError),
        (FedErrorReason.BINARY_TARGET_INVALID, ValueError),
        (FedErrorReason.CATEGORICAL_PASSTHROUGH_UNSUPPORTED, ValueError),
        (FedErrorReason.COX_RISK_SET_INVALID, ValueError),
        (FedErrorReason.GLMM_LAPLACE_REQUIRED, ValueError),
        (FedErrorReason.GLMM_VARIANCE_POSITIVE_REQUIRED, ValueError),
        (FedErrorReason.HISTOGRAM_ARRAY_DIMENSIONS_INVALID, ValueError),
        (FedErrorReason.INTERVAL_INVALID, ValueError),
        (FedErrorReason.KFOLD_SETTINGS_INVALID, ValueError),
        (FedErrorReason.KMEANS_FEATURE_MISMATCH, ValueError),
        (FedErrorReason.KMEANS_INITIALIZATION_UNSUPPORTED, ValueError),
        (FedErrorReason.LMM_INITIAL_VARIANCE_INVALID, ValueError),
        (FedErrorReason.MIXED_EFFECTS_WEIGHTS_INVALID, ValueError),
        (FedErrorReason.ORDINAL_TARGET_RANGE_INVALID, ValueError),
        (FedErrorReason.OUTLIER_STRATEGY_UNSUPPORTED, RuntimeError),
        (FedErrorReason.PAIRED_SAMPLE_LENGTH_MISMATCH, ValueError),
        (FedErrorReason.PARAMETER_CHOICE_INVALID, ValueError),
        (FedErrorReason.PARAMETER_CONVERSION_INVALID, ValueError),
        (FedErrorReason.PARAMETER_RANGE_INVALID, ValueError),
        (FedErrorReason.PARAMETER_TYPE_INVALID, TypeError),
        (FedErrorReason.REGRESSION_OUTCOME_VARIANCE_INVALID, ValueError),
        (FedErrorReason.REGRESSION_TEST_SET_EMPTY, ValueError),
        (FedErrorReason.SURVIVAL_TARGET_SHAPE_INVALID, ValueError),
        (FedErrorReason.TIME_BINS_ORDER_INVALID, ValueError),
        (FedErrorReason.EVENT_TIMES_ORDER_INVALID, ValueError),
        (FedErrorReason.TRANSFORMER_OUTPUT_NONNUMERIC, TypeError),
    ],
)
def test_internal_contract_failures_use_native_exceptions(reason, target):
    fields = {
        name: 0
        for _, name, _, _ in Formatter().parse(reason.template)
        if name is not None
    }
    error = FedValueError(reason, **fields)

    translated = translate_federated_error(error)

    assert type(translated) is target
    assert str(translated) == str(error)


def test_every_reason_has_an_explicit_exareme3_policy():
    assert set(_TRANSLATIONS) == set(FedErrorReason)
    assert all(
        issubclass(exception_type, Exception)
        and (policy is KEEP_MESSAGE or isinstance(policy, str) and policy.strip())
        for exception_type, policy in _TRANSLATIONS.values()
    )


def test_reason_values_derive_from_names():
    for reason in FedErrorReason:
        assert reason.value == reason.name.lower()
        assert FedErrorReason(reason.value) is reason


def test_reasons_with_identical_templates_remain_distinct():
    observations = FedErrorReason.INSUFFICIENT_OBSERVATIONS
    samples = FedErrorReason.INSUFFICIENT_SAMPLE_COUNT
    assert observations.template == samples.template
    assert observations is not samples


@pytest.mark.parametrize("reason", list(FedErrorReason))
def test_every_reason_uses_its_declared_policy(reason):
    fields = {
        name: 0
        for _, name, _, _ in Formatter().parse(reason.template)
        if name is not None
    }
    error = FedValueError(reason, **fields)
    exception_type, policy = _TRANSLATIONS[reason]
    expected = str(error) if policy is KEEP_MESSAGE else policy

    translated = translate_federated_error(error)

    assert type(translated) is exception_type
    assert str(translated) == expected


@pytest.mark.parametrize("reason", [None, "insufficient_fold_samples", 1, object()])
def test_fed_error_rejects_non_enum_reasons(reason):
    with pytest.raises(FedTypeError, match="FedErrorReason") as exc_info:
        FedValueError(reason)
    assert exc_info.value.reason is FedErrorReason.INVALID_ERROR_REASON


def test_fed_error_requires_a_reason():
    with pytest.raises(TypeError):
        FedValueError()


def test_missing_diagnostic_fields_fail_immediately():
    with pytest.raises(KeyError, match="n_samples"):
        FedValueError(FedErrorReason.INSUFFICIENT_FOLD_SAMPLES)


def test_diagnostic_details_preserve_counts_and_message():
    error = FedValueError(
        FedErrorReason.INSUFFICIENT_FOLD_SAMPLES, n_samples=7, n_splits=8
    )

    assert str(error) == (
        "Cross validation cannot run because the number of observations (7) "
        "is smaller than the number of splits (8)."
    )
    assert error.message == str(error)
    assert error.details == {"n_samples": 7, "n_splits": 8}
    assert "(7)" not in translate_federated_error(error).message


def test_logistic_sample_count_message_passes_through():
    with pytest.raises(FedError) as exc_info:
        FederatedLogisticRegression._handle_logreg_errors(nobs=2, p=2, y_sum=1)

    assert exc_info.value.reason is FedErrorReason.INSUFFICIENT_SAMPLE_COUNT
    assert translate_federated_error(exc_info.value).message == (
        "Logistic regression requires more than 2 observations. "
        "Select more observations or fewer covariates."
    )
