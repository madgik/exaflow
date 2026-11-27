"""Standalone algorithm exceptions and their diagnostic reasons."""

from enum import Enum
from enum import auto
from enum import unique


def error_reason(template: str) -> tuple[auto, str]:
    """Declare a diagnostic template with an automatically named enum value."""
    return auto(), template


@unique
class FedErrorReason(Enum):
    """Stable failure identities and library diagnostic templates."""

    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()

    ANOVA_FACTOR_LEVELS_INSUFFICIENT = error_reason(
        "The variable {variable} has less than 2 levels and Anova cannot be "
        "performed. Please choose another variable.",
    )
    ANOVA_GROUP_LABEL_MISMATCH = error_reason(
        "Provide one category label for each ANOVA group.",
    )
    ANOVA_OBSERVED_LEVELS_INSUFFICIENT = error_reason(
        "The data of variable {variable} contain less than 2 levels and Anova "
        "cannot be performed. Please select more data or choose another variable.",
    )
    ARRAY_DIMENSIONS_INVALID = error_reason(
        "{input_name} must be an array with {ranks} dimensions.",
    )
    ARRAY_SHAPES_MISMATCH = error_reason(
        "Arrays must have the same shape",
    )
    BINARY_TARGET_INVALID = error_reason(
        "Outcome values must be encoded as 0 or 1.",
    )
    CATEGORICAL_NB_NOT_FITTED = error_reason(
        "CategoricalNB is not fitted yet.",
    )
    CATEGORICAL_PASSTHROUGH_UNSUPPORTED = error_reason(
        "FederatedColumnTransformer remainder passthrough does not support "
        "categorical vars: {remaining_cat}",
    )
    CATEGORY_ORDER_INVALID = error_reason(
        "Category ordering must contain unique, non-missing category values.",
    )
    CATEGORY_ORDER_REQUIRED = error_reason(
        "Category ordering must be provided explicitly.",
    )
    CLASSIFICATION_LABELS_REQUIRED = error_reason(
        "Provide outcome labels for fitting.",
    )
    CLASSIFICATION_LABEL_DIMENSIONS_INVALID = error_reason(
        "Outcome must be a one-dimensional array of class labels.",
    )
    COLUMN_MASK_LENGTH_INVALID = error_reason(
        "Boolean column mask length must match number of columns.",
    )
    COX_EVENTS_REQUIRED = error_reason(
        "cox_regression_classical requires at least one observed event.",
    )
    COX_GLOBAL_EVENTS_REQUIRED = error_reason(
        "cox_regression_classical requires at least one observed event globally.",
    )
    COX_RISK_SET_INVALID = error_reason(
        "Encountered an empty or invalid risk set while fitting "
        "cox_regression_classical.",
    )
    CV_FEATURE_VARIABLES_REQUIRED = error_reason(
        "categorical_vars and numerical_vars are required.",
    )
    CV_INPUT_REQUIRED = error_reason(
        "X is required when data is not provided.",
    )
    CV_SPLITTER_INTERFACE_INVALID = error_reason(
        "Splitter must support split_indices for DataFrame CV.",
    )
    DATAFRAME_COLUMN_MISSING = error_reason(
        "Input data is missing one or more required columns.",
    )
    DATAFRAME_REQUIRED = error_reason(
        "Provide the input data as a pandas DataFrame.",
    )
    DATA_ARRAYS_EMPTY = error_reason(
        "Data arrays cannot be empty",
    )
    DATA_ARRAY_EMPTY = error_reason(
        "Data array cannot be empty",
    )
    DDOF_OBSERVATIONS_INSUFFICIENT = error_reason(
        "Not enough data points for ddof={ddof}. Need at least {minimum} points.",
    )
    EMPTY_CONTINGENCY_TABLE = error_reason(
        "Contingency table is empty after pruning.",
    )
    ENCODED_FEATURES_INVALID = error_reason(
        "Encoded features must be a rectangular array.",
    )
    ENCODER_CATEGORIES_REQUIRED = error_reason(
        "FederatedOrdinalEncoder requires categories to be provided via the "
        "constructor.",
    )
    ENCODER_UNKNOWN_CATEGORY = error_reason(
        "Unknown categories encountered in column '{var}'.",
    )
    EVENT_TIMES_EMPTY = error_reason(
        "event_times must contain at least one failure time.",
    )
    EVENT_TIMES_ORDER_INVALID = error_reason(
        "event_times must be strictly increasing.",
    )
    EVENT_TIMES_POSITIVE_REQUIRED = error_reason(
        "event_times must be strictly positive.",
    )
    EVENT_TIME_INTERVAL_EMPTY = error_reason(
        "event_times must define at least one failure time.",
    )
    FEATURE_NAMES_MISMATCH = error_reason(
        "Provide one feature name for each feature column.",
    )
    FEATURE_SCHEMA_MISMATCH = error_reason(
        "Prediction features must match the fitted model. Provide the same "
        "features in the same order used for fitting.",
    )
    FISHER_TABLE_SHAPE = error_reason(
        "Expected a 2x2 contingency table, got shape {shape}.",
    )
    GAUSSIAN_NB_NOT_FITTED = error_reason(
        "GaussianNB is not fitted yet.",
    )
    GLMM_BINARY_TARGET_INVALID = error_reason(
        "GLMM binary expects y in {{0,1}}.",
    )
    GLMM_LAPLACE_CONFIGURATION_INVALID = error_reason(
        "FederatedGLMMBinary requires add_laplace_corrections=True to estimate "
        "sigma_u2 correctly.",
    )
    GLMM_LAPLACE_REQUIRED = error_reason(
        "GLMM binary requires add_laplace_corrections=True.",
    )
    GLMM_PAYLOAD_INVALID = error_reason(
        "payload must contain 'score' and 'h_packed'",
    )
    GLMM_SCORE_LENGTH_INVALID = error_reason(
        "score length {score_length} != q={q}",
    )
    GLMM_THETA_LENGTH_INVALID = error_reason(
        "theta length {theta_length} != q={q}",
    )
    GLMM_THETA_SHAPE_INVALID = error_reason(
        "theta length mismatch.",
    )
    GLMM_VARIANCE_POSITIVE_REQUIRED = error_reason(
        "sigma_u2 must be positive.",
    )
    GROUP_COUNT_INVALID = error_reason(
        "n_groups must be positive.",
    )
    HISTOGRAM_ARRAY_DIMENSIONS_INVALID = error_reason(
        "Input 'x' must be a 1D array.",
    )
    HISTOGRAM_FINITE_VALUES_REQUIRED = error_reason(
        "No finite data available to compute histogram.",
    )
    INCONSISTENT_LENGTH = error_reason(
        "Provide one outcome value for each observation in the feature matrix.",
    )
    INSUFFICIENT_ANOVA_LEVELS = error_reason(
        "Cannot perform Anova one-way. Covariable has only one level.",
    )
    INSUFFICIENT_FOLD_SAMPLES = error_reason(
        "Cross validation cannot run because the number of observations "
        "({n_samples}) is smaller than the number of splits ({n_splits}).",
    )
    INSUFFICIENT_OBSERVATIONS = error_reason(
        "{name} requires {comparison} {minimum} observations. Select more "
        "observations or fewer covariates.",
    )
    INSUFFICIENT_SAMPLE_COUNT = error_reason(
        "{name} requires {comparison} {minimum} observations. Select more "
        "observations or fewer covariates.",
    )
    INTERVAL_INVALID = error_reason(
        "{name} must contain two ordered limits without NaN.",
    )
    INVALID_ERROR_REASON = error_reason(
        "reason must be a FedErrorReason member.",
    )
    KFOLD_SETTINGS_INVALID = error_reason(
        "Provide valid K-fold settings. Set random_state only when shuffle is enabled.",
    )
    KMEANS_CENTERS_REQUIRED = error_reason(
        "At least one two-dimensional center is required.",
    )
    KMEANS_CENTERS_FINITE_VALUES_REQUIRED = error_reason(
        "K-means cluster centers must contain only finite values.",
    )
    KMEANS_CLUSTER_RANGE_INVALID = error_reason(
        "k_max must be greater than or equal to k_min.",
    )
    KMEANS_FEATURE_MISMATCH = error_reason(
        "Observations and centers must have matching features.",
    )
    KMEANS_FINITE_VALUES_REQUIRED = error_reason(
        "K-means requires finite numerical values.",
    )
    KMEANS_FIT_FINITE_VALUES_REQUIRED = error_reason(
        "K-means requires finite numerical values. Apply missing-value handling "
        "and remove infinite values before fitting.",
    )
    KMEANS_INITIALIZATIONS_INVALID = error_reason(
        "n_init must be greater than or equal to 1.",
    )
    KMEANS_INITIALIZATION_UNSUPPORTED = error_reason(
        "Unsupported KMeans initialization method: '{init_method}'.",
    )
    KMEANS_MIN_CLUSTERS_INVALID = error_reason(
        "k_min must be greater than or equal to 1.",
    )
    LMM_INITIAL_VARIANCE_INVALID = error_reason(
        "init_sigma2 and init_sigma_u2 must be within [lower_bound, upper_bound].",
    )
    LMM_ITERATION_RANGE_INVALID = error_reason(
        "min_iter cannot be greater than max_iter.",
    )
    LMM_OBSERVATIONS_INSUFFICIENT = error_reason(
        "LMM cannot run because observations are fewer than predictors.",
    )
    LMM_PROFILE_STATE_MISSING = error_reason(
        "LMM finished without a profile state.",
    )
    LOGISTIC_CONVERGENCE_FAILED = error_reason(
        "Logistic regression cannot converge. Cancelling run.",
    )
    MANN_WHITNEY_GROUP_EMPTY = error_reason(
        "Both groups must have at least one observation.",
    )
    MIXED_EFFECTS_GROUP_SIZE_INVALID = error_reason(
        "nj must be positive.",
    )
    MIXED_EFFECTS_VARIANCES_INVALID = error_reason(
        "Require sigma2>0 and sigma_u2>=0.",
    )
    MIXED_EFFECTS_WEIGHTS_INVALID = error_reason(
        "w must be non-negative and finite.",
    )
    MODEL_NOT_FITTED = error_reason(
        "This model is not fitted yet. Fit it before prediction or transformation.",
    )
    NAMED_STEP_INVALID = error_reason(
        "Provide each component as a named step with the required arguments.",
    )
    NB_FEATURE_SCHEMA_INVALID = error_reason(
        "Features must match the configured feature columns.",
    )
    NUMERIC_ARRAY_REQUIRED = error_reason(
        "{input_name} must be a numeric array with {dimensions} "
        "dimensions{finite_requirement}.",
    )
    NUMERIC_VALUES_REQUIRED = error_reason(
        "{input_name} must contain numeric values.",
    )
    NUMPY_ARRAYS_REQUIRED = error_reason(
        "Inputs must be numpy arrays",
    )
    NUMPY_ARRAY_REQUIRED = error_reason(
        "Input must be a numpy array",
    )
    ONE_DIMENSIONAL_ARRAYS_REQUIRED = error_reason(
        "Inputs must be 1D arrays",
    )
    ONE_DIMENSIONAL_ARRAY_REQUIRED = error_reason(
        "Input must be a 1D array",
    )
    ORDINAL_CATEGORY_COUNT_INVALID = error_reason(
        "K must be >= 2.",
    )
    ORDINAL_CATEGORY_COUNT_VALUE_INVALID = error_reason(
        "K must be >= 2 (got {K}).",
    )
    ORDINAL_TARGET_LENGTH_INVALID = error_reason(
        "eta and y must have same length.",
    )
    ORDINAL_TARGET_RANGE_INVALID = error_reason(
        "y must be in [0, {max_category}].",
    )
    ORDINAL_THETA_LENGTH_INVALID = error_reason(
        "theta length {theta_length} != expected {expected_len} (p={p}, K={K}, "
        "fit_intercept={fit_intercept}).",
    )
    ORDINAL_VARIANCE_POSITIVE_REQUIRED = error_reason(
        "sigma_u2 must be > 0.",
    )
    OUTLIER_FOLD_FINITE_REQUIRED = error_reason(
        "Fold should be finite.",
    )
    OUTLIER_FOLD_POSITIVE_REQUIRED = error_reason(
        "Fold should be greater than 0.",
    )
    OUTLIER_OBSERVATIONS_INSUFFICIENT = error_reason(
        "Insufficient non-missing data for variable '{variable}': {num_dtps} rows;"
        " minimum required is {min_row_count}.",
    )
    OUTLIER_QUANTILE_FOLD_INVALID = error_reason(
        "Quantile fold should be greater than 0 and less than 0.5.",
    )
    OUTLIER_STRATEGY_UNSUPPORTED = error_reason(
        "Unsupported outlier strategy '{strategy}'.",
    )
    PAIRED_SAMPLE_LENGTH_MISMATCH = error_reason(
        "Paired samples must have the same length.",
    )
    PARAMETER_CHOICE_INVALID = error_reason(
        "{name} must be one of: {options}.",
    )
    PARAMETER_CONVERSION_INVALID = error_reason(
        "{name} must be a valid {type_name} value.",
    )
    PARAMETER_RANGE_INVALID = error_reason(
        "{name} must be a {kind}{bounds}.",
    )
    PARAMETER_TYPE_INVALID = error_reason(
        "{name} must be {description}.",
    )
    PCA_COMPONENT_DIMENSIONS_INVALID = error_reason(
        "components must be a two-dimensional array.",
    )
    PEARSON_DATA_EMPTY = error_reason(
        "Cannot compute Pearson correlation on empty data.",
    )
    PEARSON_OBSERVATIONS_INSUFFICIENT = error_reason(
        "Not enough observations to compute Pearson correlation.",
    )
    PIPELINE_INTERMEDIATE_TYPE_INVALID = error_reason(
        "Pipeline step outputs ndarray before final step. Only the last step may "
        "output an array.",
    )
    PIPELINE_OUTPUT_TYPE_INVALID = error_reason(
        "Pipeline transformers must return ndarray.",
    )
    PIPELINE_TARGET_REQUIRED = error_reason(
        "Pipeline estimator fit requires y.",
    )
    PIPELINE_TRANSFORMER_REQUIRED = error_reason(
        "Pipeline step '{name}' is not a transformer.",
    )
    REGRESSION_OUTCOME_VARIANCE_INVALID = error_reason(
        "Aggregated test outcome variance is negative.",
    )
    REGRESSION_TEST_SET_EMPTY = error_reason(
        "Regression scoring requires a nonempty global test set.",
    )
    SCORER_LABELS_REQUIRED = error_reason(
        "Labels must be provided or available on estimator results.",
    )
    SMD_GROUP_OBSERVATIONS_INSUFFICIENT = error_reason(
        "Each group needs at least 2 points for SMD",
    )
    SQL_AGGREGATES_REQUIRED = error_reason(
        "At least one aggregate() must be defined before run().",
    )
    SQL_QUERY_ALREADY_EXECUTED = error_reason(
        "run() can only be executed once per query builder object.",
    )
    STACKED_COX_EVENTS_INSUFFICIENT = error_reason(
        "cox_regression_stacked requires more observed events than covariates to "
        "fit the stacked logistic model.",
    )
    STACKED_COX_EVENTS_REQUIRED = error_reason(
        "cox_regression_stacked requires at least one observed event.",
    )
    STACKED_COX_GLOBAL_EVENTS_REQUIRED = error_reason(
        "cox_regression_stacked requires at least one observed event globally.",
    )
    STEP_NAMES_DUPLICATED = error_reason(
        "Component names must be unique.",
    )
    STEP_NAME_INVALID = error_reason(
        "Each component must have a nonempty string name.",
    )
    SURVIVAL_COVARIATE_REQUIRED = error_reason(
        "{name} requires at least one covariate.",
    )
    SURVIVAL_OBSERVATIONS_REQUIRED = error_reason(
        "{name} requires at least one observation.",
    )
    SURVIVAL_TARGET_EMPTY = error_reason(
        "Survival target y must contain at least one observation.",
    )
    SURVIVAL_TARGET_SHAPE_INVALID = error_reason(
        "Survival target y must have shape (n_obs, 2) with columns [time, event].",
    )
    TIMES_POSITIVE_REQUIRED = error_reason(
        "All time values must be strictly positive.",
    )
    TIME_BINS_EMPTY = error_reason(
        "time_bins must define at least one interval.",
    )
    TIME_BINS_END_INVALID = error_reason(
        "The final time bin cut point must be greater than the maximum time value.",
    )
    TIME_BINS_ORDER_INVALID = error_reason(
        "time_bins must be strictly increasing.",
    )
    TIME_BINS_START_INVALID = error_reason(
        "time_bins must start at 0.",
    )
    TIME_BINS_TOO_SHORT = error_reason(
        "time_bins must contain at least two cut points.",
    )
    TIME_GRID_EMPTY = error_reason(
        "time grid must contain at least one value.",
    )
    TRANSFORMER_OUTPUT_NONNUMERIC = error_reason(
        "Transformer output must contain numeric values.",
    )
    TTEST_GROUP_A_EMPTY = error_reason(
        "Group A has no data.",
    )
    TTEST_GROUP_B_EMPTY = error_reason(
        "Group B has no data.",
    )
    TTEST_INDEPENDENT_OBSERVATIONS_INSUFFICIENT = error_reason(
        "Not enough observations for independent t-test.",
    )
    TTEST_ONESAMPLE_OBSERVATIONS_INSUFFICIENT = error_reason(
        "Not enough observations for one-sample t-test.",
    )
    TTEST_PAIRED_OBSERVATIONS_INSUFFICIENT = error_reason(
        "Not enough observations for paired t-test.",
    )
    ZERO_RANK_VARIATION = error_reason(
        "Standard deviation is zero; cannot compute z-score.",
    )

    def __new__(cls, code: str, template: str):
        member = object.__new__(cls)
        member._value_ = code
        member.template = template
        return member


class FedError(Exception):
    """A declared algorithm failure with a mandatory, structured reason."""

    def __init__(self, reason: FedErrorReason, **details):
        if not isinstance(reason, FedErrorReason):
            raise FedTypeError(FedErrorReason.INVALID_ERROR_REASON)
        self.reason = reason
        self.details = details
        self.message = reason.template.format(**details)
        super().__init__(self.message)


class FedValueError(FedError, ValueError):
    """An argument value or statistical prerequisite is invalid."""


class FedTypeError(FedError, TypeError):
    """An argument or component has an unsupported type."""


class FedStateError(FedError, RuntimeError):
    """An operation is invalid in the current computation state."""


class FedObservationError(FedValueError):
    """The observations do not satisfy the computation's requirements."""
