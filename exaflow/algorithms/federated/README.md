# Federated Algorithms

Implementations are grouped by domain. Encoders live in `feature_engineering`,
column composition and passthrough in `compose`, and outlier reporting with its
winsorization helpers in `outliers`. Cross-validation machinery remains under
`model_selection/cross_validation` because it serves multiple model families.
Mixed-effects numerical helpers live in `mixed_effects/mixed_effects_utils.py`.

## Input validation

The `validation` package provides reusable parameter type rules, numeric-array
policies, row/feature checks, binary-target checks, and sample-count checks. It
uses public scikit-learn validation helpers without requiring estimator
inheritance or importing controller/worker services. Algorithm integrations keep
their existing normalization and method-specific mathematical requirements. See
the [validation coverage and error conventions](docs/input_validation.md) for the
complete inventory, retained behaviors, and Exareme3 boundary changes.

Apply each method's existing normalization before checking its effective inputs.
Do not introduce uniform restrictions on missing values, empty partitions,
feature counts, or class availability. Preserve method-specific behavior.

Validators raise `BadInputError` locally without performing aggregation. The
caller owns error propagation, cancellation, and cleanup of the distributed
computation. Exareme3 handles request failure and aggregation cleanup through
its existing worker/controller error handling; it does not guarantee immediate
cancellation of every running participant. Independent callers must provide
their own lifecycle handling. Participants still require the same model
configuration and aligned feature layout for the mathematical aggregations.

Validation unit tests live in `tests/standalone_tests/federated_validation/`.
Existing algorithm/production tests and fixtures remain unchanged.

## cluster

- [KMEANS](docs/kmeans.md)

## compose

- [COLUMN_TRANSFORMER](docs/column_transformer.md)
- [MAKE_COLUMN_SELECTOR](docs/make_column_selector.md)
- [PASSTHROUGH](docs/passthrough.md)

## decomposition

- [PCA](docs/pca.md)

## linear_model

- [COX_REGRESSION_CLASSICAL](docs/cox_regression_classical.md)
- [COX_REGRESSION_STACKED](docs/cox_regression_stacked.md)
- [LOGISTIC_REGRESSION](docs/logistic_regression.md)
- [OLS](docs/ols.md)

## mixed_effects

- [LMM](docs/lmm.md)
- [GLMM_BINARY](docs/glmm_binary.md)
- [GLMM_ORDINAL](docs/glmm_ordinal.md)

## model_selection

- [CROSS_VALIDATOR](docs/cross_validator.md)
- [KFOLD_SPLITTER](docs/kfold_splitter.md)
- [SCORERS](docs/scorers.md)

## naive_bayes

- [GAUSSIAN_NAIVE_BAYES](docs/gaussian_naive_bayes.md)
- [CATEGORICAL_NAIVE_BAYES](docs/categorical_naive_bayes.md)

## feature_engineering

- [ONE_HOT_ENCODER](docs/one_hot_encoder.md)
- [ORDINAL_ENCODER](docs/ordinal_encoder.md)

## outliers

- [OUTLIER_REPORT](docs/outlier_report.md)

## statistics

- [DESCRIBE](docs/describe.md)
- [DESCRIPTIVE_STATISTICS](docs/descriptive_statistics.md)
- [PEARSON_CORRELATION](docs/pearson_correlation.md)
- [TTEST_INDEPENDENT](docs/ttest_independent.md)
- [TTEST_ONE_SAMPLE](docs/ttest_one_sample.md)
- [TTEST_PAIRED](docs/ttest_paired.md)
- [ANOVA_ONE_WAY](docs/anova_one_way.md)
- [ANOVA_TWO_WAY](docs/anova_two_way.md)
- [HISTOGRAM](docs/histogram.md)
- [BINNED_MANN_WHITNEY_U_TEST](docs/binned_mann_whitney_u_test.md)
- [STANDARDIZED_MEAN_DIFFERENCE](docs/standardized_mean_difference.md)
