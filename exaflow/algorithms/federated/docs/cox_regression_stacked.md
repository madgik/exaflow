# Stacked survival regression

Implementation: [linear_model/cox_regression_stacked.py](../linear_model/cox_regression_stacked.py).

Stacked Cox regression approximates time-to-event modeling by expanding survival
data into risk-set rows and fitting a logistic regression model with time-bin
indicators. It reports covariate coefficients, approximate hazard ratios, and
classification-style model summaries from the stacked fit.

## Library interface

`FederatedStackedCoxRegression(time_grid_strategy="distinct_event_times", n_time_bins=10).fit(X, y, *, agg_client, feature_names=None)` takes numeric features and a two-column target: follow-up time and binary event indicator. Constructor settings are keyword-only. Results contain covariate coefficients and approximate hazard ratios; nuisance time-bin coefficients are not covariate effects.

## Statistical method

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
Receive numeric X and y[:, 0] = time, y[:, 1] = binary event.
Validate survival inputs and aggregate event/sample requirements.
Build a common grid from distinct event times or uniform intervals.
Expand each local observation into eligible risk-set time rows.
Include time-bin indicators as nuisance baseline terms.
Fit the expanded binary model using the shared logistic estimator.
Return covariate effects, approximate hazard ratios, and grid diagnostics.
```

## Numerical decisions

- The default time grid uses distinct observed event times.
- Uniform time bins start at zero and end above the maximum follow-up time.
- The number of time bins is capped when needed for identifiability.
- Time-bin indicator coefficients are used internally and not reported as
  covariate effects.
- This method is not identical to Cox partial likelihood.

## Reference comparison

Standalone Cox tests include comparisons with a centralized
`statsmodels.duration.hazard_regression.PHReg` reference for Cox methodology.
The stacked model should be described as methodologically related to
discrete-time survival stacking, not exactly equivalent to `PHReg`.

## Limitations

- Follow-up times must be strictly positive.
- At least one observed event is required.
- More observed events than covariates are required for an identifiable stacked
  logistic model.
- Results depend on the chosen time grid.
- Time-varying covariates are not supported.
- Hazard ratios are approximate because the model is fitted through survival
  stacking rather than exact Cox partial likelihood.
