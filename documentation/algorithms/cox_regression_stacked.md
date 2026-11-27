# Stacked Cox Regression

## Overview

Stacked Cox regression approximates time-to-event modeling by expanding survival
data into risk-set rows and fitting a logistic regression model with time-bin
indicators. It reports covariate coefficients, approximate hazard ratios, and
classification-style model summaries from the stacked fit.

## Inputs

### Required inputs

| Input | Description |
|---|---|
| `y` | Positive numerical follow-up time. |
| `x` | Event variable plus one or more covariates. |

### Parameters

| Parameter | Description | Default |
|---|---|---|
| `event_var` | Variable from `x` used to build the binary event vector. | Required |
| `positive_class` | Event level mapped to `1`; other observed levels map to `0`. | Optional |
| `time_grid_strategy` | Time discretization strategy for survival stacking. | `distinct_event_times` |
| `n_time_bins` | Bin count used only with the `uniform` time grid strategy. | `10` |

## Method

The mathematical method, library API, aggregation flow, numerical decisions, reference comparisons, and limitations are documented in [Stacked survival regression](../../exaflow/algorithms/federated/docs/cox_regression_stacked.md).

## Application integration

The wrapper interprets `0/1` and Boolean event variables automatically; other categorical event variables require `positive_class`. It one-hot encodes categorical covariates and constructs the library target as `[time, event]`. Missing-value preprocessing precedes fitting.

## Outputs

| Field | Description |
|---|---|
| `dependent_var` | Follow-up time variable. |
| `event_var` | Event variable used for event coding. |
| `indep_vars` | Covariate names after encoding. |
| `summary.n_obs` | Number of original observations. |
| `summary.n_events` | Number of observed events. |
| `summary.n_stacked_rows` | Number of expanded risk-set rows. |
| `summary.coefficients` | Covariate coefficients from the stacked model. |
| `summary.hazard_ratios` | Exponentiated covariate coefficients. |
| `summary.std_err` | Standard errors. |
| `summary.z_scores`, `summary.pvalues` | Wald test statistics and p-values. |
| `summary.r_squared_cs`, `summary.r_squared_mcf` | Pseudo R-squared values. |
| `summary.ll0`, `summary.ll` | Null and fitted log-likelihoods. |
| `summary.aic`, `summary.bic` | Information criteria. |
| `summary.time_grid_strategy` | Time-grid strategy used. |
| `summary.n_time_bins_used` | Number of time bins in the stacked design. |
