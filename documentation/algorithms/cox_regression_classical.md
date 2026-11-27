# Cox Proportional Hazards Regression

## Overview

Cox proportional hazards regression models time-to-event data using the partial
likelihood. It estimates covariate effects on the hazard while leaving the
baseline hazard unspecified.

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

## Method

The mathematical method, library API, aggregation flow, numerical decisions, reference comparisons, and limitations are documented in [Classical Cox regression](../../exaflow/algorithms/federated/docs/cox_regression_classical.md).

## Application integration

The wrapper interprets `0/1` and Boolean event variables automatically; other categorical event variables require `positive_class`. It one-hot encodes categorical covariates and constructs the library target as `[time, event]`. Missing-value preprocessing precedes fitting.

## Outputs

| Field | Description |
|---|---|
| `dependent_var` | Follow-up time variable. |
| `event_var` | Event variable used for event coding. |
| `indep_vars` | Covariate names after encoding. |
| `summary.n_obs` | Number of observations. |
| `summary.n_events` | Number of observed events. |
| `summary.n_unique_event_times` | Number of event times used in the partial likelihood. |
| `summary.coefficients` | Estimated log-hazard coefficients. |
| `summary.hazard_ratios` | Exponentiated coefficients. |
| `summary.std_err` | Standard errors. |
| `summary.lower_ci`, `summary.upper_ci` | Coefficient confidence intervals. |
| `summary.hr_lower_ci`, `summary.hr_upper_ci` | Hazard-ratio confidence intervals. |
| `summary.z_scores`, `summary.pvalues` | Wald test statistics and p-values. |
| `summary.ll` | Partial log-likelihood. |
| `summary.converged` | Convergence indicator. |
| `summary.score_norm`, `summary.step_norm` | Convergence diagnostics. |
