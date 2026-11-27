# Logistic Regression

Runtime wrapper: [linear_model/logistic_regression.py](../../exaflow/algorithms/exareme3/linear_model/logistic_regression.py).

The CV wrapper is colocated at `linear_model/logistic_regression_cv.py`.
Shared federated cross-validation utilities remain under `model_selection`.

## Overview

Logistic regression models a binary outcome as a function of numerical and/or
categorical covariates. The selected positive class is encoded as `1`; all other
outcome classes are encoded as `0`.

## Inputs

### Required inputs

| Input | Description |
|---|---|
| `y` | Nominal outcome variable converted to binary using `positive_class`. |
| `x` | One or more numerical or categorical covariates. |

### Parameters

| Parameter | Description | Default |
|---|---|---|
| `positive_class` | Outcome category treated as the positive outcome. | Required |

## Method

The mathematical method, library API, aggregation flow, numerical decisions, reference comparisons, and limitations are documented in [Binary logistic regression](../../exaflow/algorithms/federated/docs/logistic_regression.md).

## Application integration

The wrapper maps the selected `positive_class` to 1 and all other outcome levels to 0, one-hot encodes categorical covariates, and always includes an intercept. Required missing-value preprocessing runs before fitting.

## Outputs

| Field | Description |
|---|---|
| `n_obs` | Number of observations used for fitting. |
| `coefficients` | Estimated coefficients including intercept. |
| `stderr` | Standard errors. |
| `lower_ci`, `upper_ci` | Confidence interval bounds. |
| `z_scores` | Wald z statistics. |
| `pvalues` | Two-sided coefficient p-values. |
| `df_model` | Model degrees of freedom. |
| `df_resid` | Residual degrees of freedom. |
| `r_squared_cs` | Cox-Snell pseudo R-squared. |
| `r_squared_mcf` | McFadden pseudo R-squared. |
| `ll0` | Null-model log-likelihood. |
| `ll` | Fitted-model log-likelihood. |
| `aic` | Akaike information criterion. |
| `bic` | Bayesian information criterion. |
