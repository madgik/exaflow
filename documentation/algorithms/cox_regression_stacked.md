## Cox Regression Stacked

<b><h4>Aggregation Server</h4></b>
This algorithm uses the aggregation server to fit a stacked logistic-regression
model across workers. Raw patient-level rows remain local.

#### Algorithm Description

`cox_regression_stacked` approximates Cox proportional hazards regression through the
survival-stacking formulation. Each worker expands right-censored survival data
into a person-period binary classification table using a shared global time
grid. A federated logistic-regression fit is then used to estimate covariate
effects. The reported covariate coefficients approximate Cox log-hazard ratios,
while the time-bin nuisance terms act as the baseline hazard.

#### Inputs

- `y`: one positive numerical follow-up time variable
- `x`: one or more covariates plus exactly one event-indicator variable
- `event_var`: parameter naming which variable from `x` is the event indicator
- `positive_class`: optional event label treated as event=1 when `event_var` is
  not already encoded as `0/1`
- `time_grid_strategy`: `distinct_event_times` (default) or `uniform`
- `n_time_bins`: number of bins for the `uniform` strategy

#### Exareme3 Notes

- The public algorithm name is `cox_regression_stacked`.
- The current implementation supports baseline covariates only.
- The returned summary exposes only covariate effects; time-bin coefficients are
  internal nuisance parameters.
- Hazard ratios are reported as `exp(beta)`.
- The `distinct_event_times` grid is the default because it tracks classical Cox
  risk sets more closely than coarse uniform binning.

<b><h4>Algorithm Implementation</b></h4>

[Cox Regression Stacked](../../exaflow/algorithms/exareme3/linear_model/cox_regression_stacked.py)
