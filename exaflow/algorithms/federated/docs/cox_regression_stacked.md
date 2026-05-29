## Cox Regression Stacked (FederatedStackedCoxRegression)

### Name

**Cox Regression Stacked (FederatedStackedCoxRegression)**

### Type

**Survival model** implemented as a federated stacked logistic-regression
estimator.

### Goal

Approximate Cox proportional hazards regression in a federated setting without
sharing raw patient-level data. The model treats time-bin effects as nuisance
baseline-hazard terms and reports only the covariate coefficients and their
derived hazard ratios.

### When to use

Use this model when:

- you need interpretable hazard-ratio style coefficients in federated survival analysis
- data are right-censored and split across workers
- baseline covariates are sufficient for the first analysis pass
- a Cox-like approximation is acceptable

### When to be careful

Be careful when:

- you require exact classical Cox partial-likelihood fitting
- the event indicator is not binary and `positive_class` is omitted
- there are too few observed events
- the chosen time grid is too coarse for the event process
- time-varying covariates are required

### Inputs / Outputs

| Item | Description |
| --- | --- |
| `X` | Local covariate matrix after federated categorical encoding |
| `times` | Positive follow-up times |
| `events` | Binary event indicators |
| `time_grid_strategy` | `distinct_event_times` or `uniform` |
| `n_time_bins` | Uniform-bin count when that strategy is selected |
| `agg_client` | Federated aggregation client |

Outputs include:

- `params`: Cox-like log-hazard ratio coefficients
- `hazard_ratios`
- `std_err`, `z_scores`, `pvalues`
- coefficient and hazard-ratio confidence intervals
- stacked-model fit statistics (`ll`, `aic`, `bic`, pseudo R²)
- `n_obs`, `n_events`, `n_stacked_rows`

### Implementation Notes

- Global distinct event times are negotiated with `agg_client.union(...)`.
- Uniform grids are available as an explicit fallback mode.
- Time-bin columns are one-hot encoded and the logistic fit runs with
  `fit_intercept=False` to avoid redundant intercept terms.
- The reported `df_resid`, `ll`, `aic`, and `bic` come from the stacked logistic
  model rather than a classical Cox partial likelihood.

### Exactness

This implementation is an approximation to Cox proportional hazards regression
through survival stacking. It is not an exact distributed partial-likelihood
solver.
