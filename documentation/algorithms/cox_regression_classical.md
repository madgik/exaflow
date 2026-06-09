# Cox Proportional Hazards Regression

## Table of contents

- [Overview](#overview)
- [Inputs](#inputs)
  - [Required inputs](#required-inputs)
  - [Parameters](#parameters)
- [Statistical model](#statistical-model)
- [Federated computation](#federated-computation)
  - [Aggregated quantities](#aggregated-quantities)
  - [Federated flow](#federated-flow)
- [Technical decisions](#technical-decisions)
- [Outputs](#outputs)
- [Validation against state-of-the-art implementation](#validation-against-state-of-the-art-implementation)
- [Limitations and assumptions](#limitations-and-assumptions)

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

## Statistical model

The Cox proportional hazards model is:

```text
h(t | x) = h_0(t) exp(x beta)
```

Coefficients are estimated by maximizing the partial log-likelihood. Tied event
times are handled with the Breslow approximation.

## Federated computation

The model is computed without sharing row-level data. Each site contributes
event-time and risk-set sufficient statistics, and Newton updates are derived
from aggregated score and information matrices.

### Aggregated quantities

| Quantity | Purpose |
|---|---|
| Global event-time grid | Align risk sets at every observed event time. |
| Event counts per event time | Compute Breslow partial likelihood terms. |
| Sum of covariates among events | Compute score contributions. |
| Risk-set sums `S0` | Denominator of the partial likelihood. |
| Risk-set weighted covariate sums `S1` | Compute score and information. |
| Risk-set weighted outer products `S2` | Compute observed information. |
| Observation and event counts | Report sample size and validate fit. |

### Federated flow

```text
Input:
    time: follow-up duration
    event_var: event indicator source
    X: covariates
    positive_class: optional event level

Step 1:
    Convert event_var to binary event values.

Step 2:
    Build the covariate matrix:
        one-hot encode categorical covariates
        include numerical covariates directly

Step 3:
    Aggregate the sorted union of observed event times.

Step 4:
    At each site:
        compute event counts and event covariate sums for each event time

Step 5:
    Aggregate static event-time quantities.

Step 6:
    Repeat Newton updates:
        each site computes risk-set S0, S1, and S2 at current coefficients
        aggregate risk-set quantities
        compute Breslow log-likelihood, score, and information
        update coefficients with a regularized or pseudo-inverse Newton step
        stop when coefficient and score criteria converge

Step 7:
    Compute hazard ratios, standard errors, Wald statistics, p-values,
    confidence intervals, and diagnostics.

Output:
    Cox regression summary
```

## Technical decisions

- Breslow handling is used for tied event times.
- Event variables stored as `0/1` or `false/true` are detected automatically.
- `positive_class` is required for non-binary categorical event variables.
- The event variable is not one-hot encoded; all other categorical covariates
  are one-hot encoded.
- Time values must be strictly positive.
- Exponentiation is clipped for numerical stability.
- Newton updates use regularization and a pseudo-inverse fallback when needed.

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

## Validation against state-of-the-art implementation

Standalone Cox tests use `statsmodels.duration.hazard_regression.PHReg` as the
centralized reference for classical Cox proportional hazards behavior.

Reference behavior is aligned with Cox partial-likelihood estimation using
Breslow tie handling.

## Limitations and assumptions

- Follow-up times must be strictly positive.
- At least one observed event is required.
- The proportional hazards assumption is not tested automatically.
- Time-varying covariates are not supported.
- Event coding must be binary after applying `positive_class`.
- High collinearity can make the information matrix unstable.
