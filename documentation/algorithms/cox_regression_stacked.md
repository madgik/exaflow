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
- `x`: the event variable plus one or more covariates
- `event_var`: parameter naming which variable from `x` is used to build the
  binary event vector
- `positive_class`: event level mapped to event `1`; all other observed levels
  are mapped to event `0`
- `time_grid_strategy`: `distinct_event_times` (default) or `uniform`
- `n_time_bins`: number of bins for the `uniform` strategy

#### Event Coding

The Cox-like stacked model is fitted with a binary event vector. If `event_var`
is stored as `0`/`1` or `false`/`true`, leave `positive_class` empty; the
algorithm detects that coding automatically.

Use `positive_class` when `event_var` is categorical, for example diagnosis
category or vital status. The selected level is converted to event `1`, and all
other observed levels are converted to event `0`. For example,
`positive_class="AD"` models AD as the event of interest and treats CN, MCI, and
other categories as event `0`.

The event variable is not one-hot encoded. Categorical covariates are one-hot
encoded before survival stacking; the event variable is converted separately.

The current specification schema can populate dropdowns from selected `y` values
or from a single selected `x` value. In this algorithm, `event_var` is selected
from a multi-variable `x` input, so `positive_class` cannot currently be exposed
as a dynamic dropdown of the selected event variable's CDE enumerations.

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
