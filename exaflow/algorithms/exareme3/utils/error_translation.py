from exaflow.algorithms.federated.exceptions import FedError
from exaflow.algorithms.federated.exceptions import FedErrorReason
from exaflow.worker_communication import AnalysisExecutionError
from exaflow.worker_communication import BadUserInput
from exaflow.worker_communication import InsufficientDataError

KEEP_MESSAGE = object()

# Each reason explicitly chooses an application exception and message policy.
_TRANSLATIONS = {
    FedErrorReason.ANOVA_FACTOR_LEVELS_INSUFFICIENT: (BadUserInput, KEEP_MESSAGE),
    FedErrorReason.ANOVA_GROUP_LABEL_MISMATCH: (ValueError, KEEP_MESSAGE),
    FedErrorReason.ANOVA_OBSERVED_LEVELS_INSUFFICIENT: (BadUserInput, KEEP_MESSAGE),
    FedErrorReason.ARRAY_DIMENSIONS_INVALID: (ValueError, KEEP_MESSAGE),
    FedErrorReason.ARRAY_SHAPES_MISMATCH: (ValueError, KEEP_MESSAGE),
    FedErrorReason.BINARY_TARGET_INVALID: (ValueError, KEEP_MESSAGE),
    FedErrorReason.CATEGORICAL_NB_NOT_FITTED: (RuntimeError, KEEP_MESSAGE),
    FedErrorReason.CATEGORICAL_PASSTHROUGH_UNSUPPORTED: (ValueError, KEEP_MESSAGE),
    FedErrorReason.CATEGORY_ORDER_INVALID: (ValueError, KEEP_MESSAGE),
    FedErrorReason.CATEGORY_ORDER_REQUIRED: (ValueError, KEEP_MESSAGE),
    FedErrorReason.CLASSIFICATION_LABELS_REQUIRED: (ValueError, KEEP_MESSAGE),
    FedErrorReason.CLASSIFICATION_LABEL_DIMENSIONS_INVALID: (ValueError, KEEP_MESSAGE),
    FedErrorReason.COLUMN_MASK_LENGTH_INVALID: (ValueError, KEEP_MESSAGE),
    FedErrorReason.COX_EVENTS_REQUIRED: (BadUserInput, KEEP_MESSAGE),
    FedErrorReason.COX_GLOBAL_EVENTS_REQUIRED: (BadUserInput, KEEP_MESSAGE),
    FedErrorReason.COX_RISK_SET_INVALID: (ValueError, KEEP_MESSAGE),
    FedErrorReason.CV_FEATURE_VARIABLES_REQUIRED: (ValueError, KEEP_MESSAGE),
    FedErrorReason.CV_INPUT_REQUIRED: (ValueError, KEEP_MESSAGE),
    FedErrorReason.CV_SPLITTER_INTERFACE_INVALID: (TypeError, KEEP_MESSAGE),
    FedErrorReason.DATAFRAME_COLUMN_MISSING: (ValueError, KEEP_MESSAGE),
    FedErrorReason.DATAFRAME_REQUIRED: (TypeError, KEEP_MESSAGE),
    FedErrorReason.DATA_ARRAYS_EMPTY: (BadUserInput, KEEP_MESSAGE),
    FedErrorReason.DATA_ARRAY_EMPTY: (BadUserInput, KEEP_MESSAGE),
    FedErrorReason.DDOF_OBSERVATIONS_INSUFFICIENT: (BadUserInput, KEEP_MESSAGE),
    FedErrorReason.EMPTY_CONTINGENCY_TABLE: (
        BadUserInput,
        (
            "There are no usable category pairs for the selected variables. Check the "
            "filters and missing-value settings."
        ),
    ),
    FedErrorReason.ENCODED_FEATURES_INVALID: (ValueError, KEEP_MESSAGE),
    FedErrorReason.ENCODER_CATEGORIES_REQUIRED: (ValueError, KEEP_MESSAGE),
    FedErrorReason.ENCODER_UNKNOWN_CATEGORY: (BadUserInput, KEEP_MESSAGE),
    FedErrorReason.EVENT_TIMES_EMPTY: (ValueError, KEEP_MESSAGE),
    FedErrorReason.EVENT_TIMES_ORDER_INVALID: (ValueError, KEEP_MESSAGE),
    FedErrorReason.EVENT_TIMES_POSITIVE_REQUIRED: (ValueError, KEEP_MESSAGE),
    FedErrorReason.EVENT_TIME_INTERVAL_EMPTY: (ValueError, KEEP_MESSAGE),
    FedErrorReason.FEATURE_NAMES_MISMATCH: (ValueError, KEEP_MESSAGE),
    FedErrorReason.FEATURE_SCHEMA_MISMATCH: (ValueError, KEEP_MESSAGE),
    FedErrorReason.FISHER_TABLE_SHAPE: (
        BadUserInput,
        (
            "Fisher's Exact Test requires exactly two observed categories in each "
            "variable. Choose two variables that meet this requirement in the selected"
            " data."
        ),
    ),
    FedErrorReason.GAUSSIAN_NB_NOT_FITTED: (RuntimeError, KEEP_MESSAGE),
    FedErrorReason.GLMM_BINARY_TARGET_INVALID: (ValueError, KEEP_MESSAGE),
    FedErrorReason.GLMM_LAPLACE_CONFIGURATION_INVALID: (ValueError, KEEP_MESSAGE),
    FedErrorReason.GLMM_LAPLACE_REQUIRED: (ValueError, KEEP_MESSAGE),
    FedErrorReason.GLMM_PAYLOAD_INVALID: (ValueError, KEEP_MESSAGE),
    FedErrorReason.GLMM_SCORE_LENGTH_INVALID: (ValueError, KEEP_MESSAGE),
    FedErrorReason.GLMM_THETA_LENGTH_INVALID: (ValueError, KEEP_MESSAGE),
    FedErrorReason.GLMM_THETA_SHAPE_INVALID: (ValueError, KEEP_MESSAGE),
    FedErrorReason.GLMM_VARIANCE_POSITIVE_REQUIRED: (ValueError, KEEP_MESSAGE),
    FedErrorReason.GROUP_COUNT_INVALID: (ValueError, KEEP_MESSAGE),
    FedErrorReason.HISTOGRAM_ARRAY_DIMENSIONS_INVALID: (ValueError, KEEP_MESSAGE),
    FedErrorReason.HISTOGRAM_FINITE_VALUES_REQUIRED: (BadUserInput, KEEP_MESSAGE),
    FedErrorReason.INCONSISTENT_LENGTH: (ValueError, KEEP_MESSAGE),
    FedErrorReason.INSUFFICIENT_ANOVA_LEVELS: (
        BadUserInput,
        (
            "One-way ANOVA requires at least two observed categories in the grouping "
            "variable. Choose a different grouping variable or adjust the filters."
        ),
    ),
    FedErrorReason.INSUFFICIENT_FOLD_SAMPLES: (
        BadUserInput,
        (
            "There are too few usable observations for the requested number of folds. "
            "Reduce the number of folds or adjust the filters."
        ),
    ),
    FedErrorReason.INSUFFICIENT_OBSERVATIONS: (
        BadUserInput,
        (
            "There are too few usable observations for this analysis. Adjust the "
            "filters or select additional data."
        ),
    ),
    FedErrorReason.INSUFFICIENT_SAMPLE_COUNT: (BadUserInput, KEEP_MESSAGE),
    FedErrorReason.INTERVAL_INVALID: (ValueError, KEEP_MESSAGE),
    FedErrorReason.INVALID_ERROR_REASON: (TypeError, KEEP_MESSAGE),
    FedErrorReason.KFOLD_SETTINGS_INVALID: (ValueError, KEEP_MESSAGE),
    FedErrorReason.KMEANS_CENTERS_FINITE_VALUES_REQUIRED: (
        AnalysisExecutionError,
        "K-means could not compute finite cluster centers.",
    ),
    FedErrorReason.KMEANS_CENTERS_REQUIRED: (ValueError, KEEP_MESSAGE),
    FedErrorReason.KMEANS_CLUSTER_RANGE_INVALID: (BadUserInput, KEEP_MESSAGE),
    FedErrorReason.KMEANS_FEATURE_MISMATCH: (ValueError, KEEP_MESSAGE),
    FedErrorReason.KMEANS_FINITE_VALUES_REQUIRED: (BadUserInput, KEEP_MESSAGE),
    FedErrorReason.KMEANS_FIT_FINITE_VALUES_REQUIRED: (BadUserInput, KEEP_MESSAGE),
    FedErrorReason.KMEANS_INITIALIZATIONS_INVALID: (ValueError, KEEP_MESSAGE),
    FedErrorReason.KMEANS_INITIALIZATION_UNSUPPORTED: (ValueError, KEEP_MESSAGE),
    FedErrorReason.KMEANS_MIN_CLUSTERS_INVALID: (ValueError, KEEP_MESSAGE),
    FedErrorReason.LMM_INITIAL_VARIANCE_INVALID: (ValueError, KEEP_MESSAGE),
    FedErrorReason.LMM_ITERATION_RANGE_INVALID: (ValueError, KEEP_MESSAGE),
    FedErrorReason.LMM_OBSERVATIONS_INSUFFICIENT: (BadUserInput, KEEP_MESSAGE),
    FedErrorReason.LMM_PROFILE_STATE_MISSING: (RuntimeError, KEEP_MESSAGE),
    FedErrorReason.LOGISTIC_CONVERGENCE_FAILED: (
        AnalysisExecutionError,
        "Logistic regression could not converge to a stable solution.",
    ),
    FedErrorReason.MANN_WHITNEY_GROUP_EMPTY: (BadUserInput, KEEP_MESSAGE),
    FedErrorReason.MIXED_EFFECTS_GROUP_SIZE_INVALID: (ValueError, KEEP_MESSAGE),
    FedErrorReason.MIXED_EFFECTS_VARIANCES_INVALID: (ValueError, KEEP_MESSAGE),
    FedErrorReason.MIXED_EFFECTS_WEIGHTS_INVALID: (ValueError, KEEP_MESSAGE),
    FedErrorReason.MODEL_NOT_FITTED: (RuntimeError, KEEP_MESSAGE),
    FedErrorReason.NAMED_STEP_INVALID: (ValueError, KEEP_MESSAGE),
    FedErrorReason.NB_FEATURE_SCHEMA_INVALID: (ValueError, KEEP_MESSAGE),
    FedErrorReason.NUMERIC_ARRAY_REQUIRED: (BadUserInput, KEEP_MESSAGE),
    FedErrorReason.NUMERIC_VALUES_REQUIRED: (BadUserInput, KEEP_MESSAGE),
    FedErrorReason.NUMPY_ARRAYS_REQUIRED: (TypeError, KEEP_MESSAGE),
    FedErrorReason.NUMPY_ARRAY_REQUIRED: (TypeError, KEEP_MESSAGE),
    FedErrorReason.ONE_DIMENSIONAL_ARRAYS_REQUIRED: (ValueError, KEEP_MESSAGE),
    FedErrorReason.ONE_DIMENSIONAL_ARRAY_REQUIRED: (ValueError, KEEP_MESSAGE),
    FedErrorReason.ORDINAL_CATEGORY_COUNT_INVALID: (ValueError, KEEP_MESSAGE),
    FedErrorReason.ORDINAL_CATEGORY_COUNT_VALUE_INVALID: (ValueError, KEEP_MESSAGE),
    FedErrorReason.ORDINAL_TARGET_LENGTH_INVALID: (ValueError, KEEP_MESSAGE),
    FedErrorReason.ORDINAL_TARGET_RANGE_INVALID: (ValueError, KEEP_MESSAGE),
    FedErrorReason.ORDINAL_THETA_LENGTH_INVALID: (ValueError, KEEP_MESSAGE),
    FedErrorReason.ORDINAL_VARIANCE_POSITIVE_REQUIRED: (ValueError, KEEP_MESSAGE),
    FedErrorReason.OUTLIER_FOLD_FINITE_REQUIRED: (BadUserInput, KEEP_MESSAGE),
    FedErrorReason.OUTLIER_FOLD_POSITIVE_REQUIRED: (BadUserInput, KEEP_MESSAGE),
    FedErrorReason.OUTLIER_OBSERVATIONS_INSUFFICIENT: (
        InsufficientDataError,
        KEEP_MESSAGE,
    ),
    FedErrorReason.OUTLIER_QUANTILE_FOLD_INVALID: (BadUserInput, KEEP_MESSAGE),
    FedErrorReason.OUTLIER_STRATEGY_UNSUPPORTED: (RuntimeError, KEEP_MESSAGE),
    FedErrorReason.PAIRED_SAMPLE_LENGTH_MISMATCH: (ValueError, KEEP_MESSAGE),
    FedErrorReason.PARAMETER_CHOICE_INVALID: (ValueError, KEEP_MESSAGE),
    FedErrorReason.PARAMETER_CONVERSION_INVALID: (ValueError, KEEP_MESSAGE),
    FedErrorReason.PARAMETER_RANGE_INVALID: (ValueError, KEEP_MESSAGE),
    FedErrorReason.PARAMETER_TYPE_INVALID: (TypeError, KEEP_MESSAGE),
    FedErrorReason.PCA_COMPONENT_DIMENSIONS_INVALID: (ValueError, KEEP_MESSAGE),
    FedErrorReason.PEARSON_DATA_EMPTY: (BadUserInput, KEEP_MESSAGE),
    FedErrorReason.PEARSON_OBSERVATIONS_INSUFFICIENT: (BadUserInput, KEEP_MESSAGE),
    FedErrorReason.PIPELINE_INTERMEDIATE_TYPE_INVALID: (TypeError, KEEP_MESSAGE),
    FedErrorReason.PIPELINE_OUTPUT_TYPE_INVALID: (TypeError, KEEP_MESSAGE),
    FedErrorReason.PIPELINE_TARGET_REQUIRED: (ValueError, KEEP_MESSAGE),
    FedErrorReason.PIPELINE_TRANSFORMER_REQUIRED: (TypeError, KEEP_MESSAGE),
    FedErrorReason.REGRESSION_OUTCOME_VARIANCE_INVALID: (ValueError, KEEP_MESSAGE),
    FedErrorReason.REGRESSION_TEST_SET_EMPTY: (ValueError, KEEP_MESSAGE),
    FedErrorReason.SCORER_LABELS_REQUIRED: (ValueError, KEEP_MESSAGE),
    FedErrorReason.SMD_GROUP_OBSERVATIONS_INSUFFICIENT: (BadUserInput, KEEP_MESSAGE),
    FedErrorReason.SQL_AGGREGATES_REQUIRED: (RuntimeError, KEEP_MESSAGE),
    FedErrorReason.SQL_QUERY_ALREADY_EXECUTED: (RuntimeError, KEEP_MESSAGE),
    FedErrorReason.STACKED_COX_EVENTS_INSUFFICIENT: (BadUserInput, KEEP_MESSAGE),
    FedErrorReason.STACKED_COX_EVENTS_REQUIRED: (BadUserInput, KEEP_MESSAGE),
    FedErrorReason.STACKED_COX_GLOBAL_EVENTS_REQUIRED: (BadUserInput, KEEP_MESSAGE),
    FedErrorReason.STEP_NAMES_DUPLICATED: (ValueError, KEEP_MESSAGE),
    FedErrorReason.STEP_NAME_INVALID: (ValueError, KEEP_MESSAGE),
    FedErrorReason.SURVIVAL_COVARIATE_REQUIRED: (BadUserInput, KEEP_MESSAGE),
    FedErrorReason.SURVIVAL_OBSERVATIONS_REQUIRED: (BadUserInput, KEEP_MESSAGE),
    FedErrorReason.SURVIVAL_TARGET_EMPTY: (BadUserInput, KEEP_MESSAGE),
    FedErrorReason.SURVIVAL_TARGET_SHAPE_INVALID: (ValueError, KEEP_MESSAGE),
    FedErrorReason.TIMES_POSITIVE_REQUIRED: (BadUserInput, KEEP_MESSAGE),
    FedErrorReason.TIME_BINS_EMPTY: (ValueError, KEEP_MESSAGE),
    FedErrorReason.TIME_BINS_END_INVALID: (ValueError, KEEP_MESSAGE),
    FedErrorReason.TIME_BINS_ORDER_INVALID: (ValueError, KEEP_MESSAGE),
    FedErrorReason.TIME_BINS_START_INVALID: (ValueError, KEEP_MESSAGE),
    FedErrorReason.TIME_BINS_TOO_SHORT: (ValueError, KEEP_MESSAGE),
    FedErrorReason.TIME_GRID_EMPTY: (ValueError, KEEP_MESSAGE),
    FedErrorReason.TRANSFORMER_OUTPUT_NONNUMERIC: (TypeError, KEEP_MESSAGE),
    FedErrorReason.TTEST_GROUP_A_EMPTY: (BadUserInput, KEEP_MESSAGE),
    FedErrorReason.TTEST_GROUP_B_EMPTY: (BadUserInput, KEEP_MESSAGE),
    FedErrorReason.TTEST_INDEPENDENT_OBSERVATIONS_INSUFFICIENT: (
        BadUserInput,
        KEEP_MESSAGE,
    ),
    FedErrorReason.TTEST_ONESAMPLE_OBSERVATIONS_INSUFFICIENT: (
        BadUserInput,
        KEEP_MESSAGE,
    ),
    FedErrorReason.TTEST_PAIRED_OBSERVATIONS_INSUFFICIENT: (BadUserInput, KEEP_MESSAGE),
    FedErrorReason.ZERO_RANK_VARIATION: (
        BadUserInput,
        (
            "The Mann–Whitney U test could not estimate enough variation in the ranks "
            "to calculate a p-value. Check that the selected values vary."
        ),
    ),
}


def translate_federated_error(error: FedError) -> Exception:
    """Apply the application exception and message policy for this reason."""
    exception_type, policy = _TRANSLATIONS[error.reason]
    message = str(error) if policy is KEEP_MESSAGE else policy
    return exception_type(message)
