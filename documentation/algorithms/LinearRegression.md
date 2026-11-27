# Linear Regression

## Overview

Linear regression fits an ordinary least squares model for a numerical outcome
using numerical and/or categorical covariates. The result includes coefficients,
standard errors, t-tests, p-values, confidence intervals, goodness-of-fit
statistics, and information criteria.

## Inputs

### Required inputs

| Input | Description |
|---|---|
| `y` | Single numerical outcome variable. |
| `x` | One or more numerical or categorical covariates. |

### Parameters

No user parameters are exposed for this algorithm.

## Method

The mathematical method, library API, aggregation flow, numerical decisions, reference comparisons, and limitations are documented in [Ordinary least squares](../../exaflow/algorithms/federated/docs/ols.md).

## Application integration

The wrapper one-hot encodes categorical covariates, passes numerical covariates through, and always includes an intercept. Required missing-value preprocessing runs before fitting. Response formatting distinguishes undefined and perfect-fit overall F statistics.

## Outputs

| Field | Description |
|---|---|
| `dependent_var` | Outcome variable name. |
| `n_obs` | Number of observations used for fitting. |
| `df_resid` | Residual degrees of freedom. |
| `df_model` | Model degrees of freedom. |
| `rse` | Residual standard error. |
| `r_squared` | R-squared. |
| `r_squared_adjusted` | Adjusted R-squared. |
| `f_stat` | Overall model F statistic. |
| `f_pvalue` | P-value for the overall F statistic. |
| `ll` | Gaussian log-likelihood. |
| `aic` | Akaike information criterion. |
| `bic` | Bayesian information criterion. |
| `indep_vars` | Feature names including the intercept. |
| `coefficients` | Estimated coefficients. |
| `std_err` | Coefficient standard errors. |
| `t_stats` | Coefficient t statistics. |
| `pvalues` | Coefficient p-values. |
| `lower_ci`, `upper_ci` | Confidence interval bounds. |
