# Linear Regression Cross-validation

Runtime wrapper: [linear_model/linear_regression_cv.py](../../exaflow/algorithms/exareme3/linear_model/linear_regression_cv.py).

The CV wrapper lives beside the regression model. Shared federated
cross-validation utilities retain their `model_selection` location.

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

## Method

See the [library method](../../exaflow/algorithms/federated/docs/ols.md) and [one-hot encoding](../../exaflow/algorithms/federated/docs/one_hot_encoder.md) for the statistical calculation and core contracts.

Shared [cross-validation](../../exaflow/algorithms/federated/docs/cross_validator.md), [K-fold splitting](../../exaflow/algorithms/federated/docs/kfold_splitter.md), and [scoring](../../exaflow/algorithms/federated/docs/scorers.md) describe fold execution and metrics.

## Application integration

The wrapper uses the requested fold count (default 5), runs the required missing-value preprocessing, and summarizes the held-out fold results. It does not return a final model refitted on all observations.

Each fold fits an intercept and learns one-hot categories from its training rows. Test-only categories become all-zero dummy columns. Metric summaries use unweighted means and sample standard deviations (`ddof=1`). If any fold metric is undefined or infinite, that metric becomes `[null, null]`; no folds are dropped and other metrics remain independently available.

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

## Compatibility

- The response field `mean_sq_error` has been removed and replaced by
  `root_mean_sq_error`. The old field already contained RMSE; no squaring or
  unit conversion is required. Consumers must use the new name and handle
  nullable metric summaries. No compatibility alias is provided.
- Deploy controller and workers together: removing the predictor-count sum
  changes the regression scoring aggregation sequence. Roll back the service
  changes and consumer field migration together. Privacy thresholds and
  dataset loading are unchanged.
