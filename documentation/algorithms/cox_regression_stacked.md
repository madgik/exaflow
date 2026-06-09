# Stacked Cox Regression

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

## Statistical model

The method uses survival stacking. For each selected time point or interval, the
risk set is expanded into binary rows indicating whether an event occurred at
that time. A logistic model is then fitted:

```text
logit(P(event at time bin | at risk, x)) =
    time_bin_effect + x beta
```

The covariate coefficients are interpreted as approximate log hazard ratios.

## Federated computation

The model is computed without sharing row-level data. Each site constructs its
own stacked risk-set rows after a shared time grid is determined. The stacked
logistic regression then uses aggregated gradients, Hessians, and
log-likelihoods.

### Aggregated quantities

| Quantity | Purpose |
|---|---|
| Maximum follow-up time | Build uniform time bins. |
| Union of observed event times | Build distinct-event-time grids. |
| Observation and event counts | Validate identifiability and report sample size. |
| Number of stacked rows | Report expanded design size. |
| Logistic gradient vector | Update stacked model coefficients. |
| Logistic Hessian matrix | Compute Newton updates and standard errors. |
| Logistic log-likelihood | Compute fit statistics and pseudo R-squared values. |

### Federated flow

```text
Input:
    time: follow-up duration
    event_var: event indicator source
    X: covariates
    time_grid_strategy: distinct_event_times or uniform
    n_time_bins: requested uniform bin count

Step 1:
    Convert event_var to binary event values.

Step 2:
    Build the covariate matrix:
        one-hot encode categorical covariates
        include numerical covariates directly

Step 3:
    Build a shared time grid:
        if uniform, aggregate maximum follow-up time
        if distinct_event_times, aggregate observed event times

Step 4:
    At each site:
        expand observations into risk-set rows
        append time-bin indicator columns
        create a binary stacked event target

Step 5:
    Fit logistic regression on the stacked design using aggregated
    gradient, Hessian, and log-likelihood terms.

Step 6:
    Retain covariate coefficients separately from time-bin effects.

Output:
    stacked Cox-style regression summary
```

## Technical decisions

- The default time grid uses distinct observed event times.
- Uniform time bins start at zero and end above the maximum follow-up time.
- The number of time bins is capped when needed for identifiability.
- Event variables stored as `0/1` or `false/true` are detected automatically.
- `positive_class` is required for non-binary categorical event variables.
- Time-bin indicator coefficients are used internally and not reported as
  covariate effects.
- This method is not identical to Cox partial likelihood.

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

## Validation against state-of-the-art implementation

Standalone Cox tests include comparisons with a centralized
`statsmodels.duration.hazard_regression.PHReg` reference for Cox methodology.
The stacked model should be described as methodologically related to
discrete-time survival stacking, not exactly equivalent to `PHReg`.

## Limitations and assumptions

- Follow-up times must be strictly positive.
- At least one observed event is required.
- More observed events than covariates are required for an identifiable stacked
  logistic model.
- Results depend on the chosen time grid.
- Time-varying covariates are not supported.
- Hazard ratios are approximate because the model is fitted through survival
  stacking rather than exact Cox partial likelihood.
