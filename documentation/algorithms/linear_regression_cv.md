# Linear Regression Cross-validation

## Overview

`linear_regression_cv` fits an ordinary least squares model on each training
fold and evaluates predictions on the corresponding held-out rows. The model
includes an intercept and one-hot encoded categorical predictors. Results
summarize RMSE, R-squared, MAE, and an F diagnostic across folds.

## Inputs

- `y`: one numerical outcome.
- `x`: numerical or categorical covariates, distinct from the outcome.
- `n_splits`: number of folds, from 2 to 20, default 5.
- Missing values must be handled by `missing_values_handler` before evaluation.

Each worker uses unshuffled K-fold splits in its local row order and must have
at least `n_splits` usable rows. Splits therefore depend on worker partitioning
and row order. They generally differ from K-fold splits of concatenated data.

## Statistical method

For each fold, let `p` be the number of encoded predictor columns, excluding
the intercept; `n_train` and `n_test` are global training and test row counts.
Predictions use the model fitted to that fold's training data.

```text
RSS = sum_test((y - prediction)^2)
TSS = sum_test((y - global_test_mean)^2)
RMSE = sqrt(RSS / n_test)
MAE = sum_test(abs(y - prediction)) / n_test
R_squared = 1 - RSS / TSS
F_diagnostic = (TSS - RSS) * (n_train - p - 1) / (p * RSS)
```

RMSE and MAE have the outcome's units. R-squared can be negative. The F
diagnostic retains the existing combination of held-out sums of squares and
training degrees of freedom. It is not a conventional fitted-model F-test,
can be negative, and has no reported significance p-value.

## Computation without row-level data

Workers align categorical levels using training data for each fold, dropping
the first sorted level as reference. They train a shared OLS model using
aggregated sufficient statistics, then aggregate held-out residual statistics.
Predictor count comes from the aligned feature matrix, not a sum of counts
across workers.

### Aggregated quantities

- Training categorical levels and OLS cross-products: `X'X`, `X'y`, and counts,
  together with the sufficient statistics used by the shared OLS implementation.
- Test residual sum of squares and absolute residual sum.
- Test row count, outcome sum, and squared-outcome sum, used to compute TSS.

### Federated flow

```text
For each local fold index:
    Split each worker's rows into training and test subsets.
    Align the training categorical schema across workers.
    Fit the global OLS model from aggregated training statistics.
    Transform test predictors using that fold's training schema.
    Predict locally and aggregate test residual/outcome statistics.
    Calculate each metric independently and record global training count.
Return identical global fold metrics from all workers.
Summarize each metric with its unweighted mean and sample standard deviation.
```

## Technical decisions

- Zero prediction error is valid and remains zero. Undefined F or R-squared
  does not replace RMSE or MAE with zeros.
- R-squared is undefined for fewer than two global test observations or no
  numerically resolvable test outcome variation.
- The F diagnostic is undefined when `p = 0`, `n_train - p - 1 <= 0`, or TSS
  is numerically zero. A perfect nonconstant fit gives an infinite diagnostic.
- Floating-point tolerances identify cancellation in TSS and effectively
  perfect fits for F. They do not round RMSE or MAE to zero.
- An empty global test set raises an explicit error. A worker may contribute
  empty test statistics when other workers have observations; the runtime's
  K-fold splitter still requires enough rows on every worker.
- If any fold value is undefined or infinite, that metric's summary is
  `[null, null]`. No folds are silently discarded and other metrics remain
  independently available.

## Outputs

| Field | Meaning |
|---|---|
| `dependent_var` | Outcome name. |
| `indep_vars` | Intercept and encoded predictor names from the complete input schema. |
| `n_obs` | Global training row count for each fold. |
| `root_mean_sq_error` | RMSE `[mean, sample standard deviation]`. |
| `r_squared` | R-squared `[mean, sample standard deviation]`. |
| `mean_abs_error` | MAE `[mean, sample standard deviation]`. |
| `f_stat` | F diagnostic `[mean, sample standard deviation]`. |

Standard deviations use `ddof=1`. Metric summaries may contain null values as
described above. The result does not include MSE, adjusted R-squared, or
residual standard error.

## Validation against a reference

Standalone tests compare predictions and held-out metrics against scikit-learn
OLS and regression metrics, using exactly the same fold membership and encoding
for one and multiple workers. See the [regression metrics documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics).
The F diagnostic is checked against its documented formula separately. Tests
also cover degenerate outcomes, intercept-only models, perfect fits, null
serialization, and the shared classification CV path.

The reference fixture generator uses centralized, unshuffled folds. Existing
fixture values are not exact distributed expectations unless partitions and
fold membership match; numerical parity is enforced by the standalone tests.

## Limitations and compatibility

- One-hot encoding uses levels observed in each training fold. A test-only
  level produces all-zero dummy columns, following the existing encoder.
- Rank-deficient models use the shared OLS pseudo-inverse. The retained F
  diagnostic uses encoded column count, not effective matrix rank.
- TSS is computed from outcome sums and squared sums, so large offsets with
  very small variation can suffer cancellation.
- The response field `mean_sq_error` has been removed and replaced by
  `root_mean_sq_error`. The old field already contained RMSE; no squaring or
  unit conversion is required. Consumers must use the new name and handle
  nullable metric summaries. No compatibility alias is provided.
- Deploy controller and workers together: removing the predictor-count sum
  changes the regression scoring aggregation sequence. Roll back the service
  changes and consumer field migration together. Privacy thresholds and
  dataset loading are unchanged.
