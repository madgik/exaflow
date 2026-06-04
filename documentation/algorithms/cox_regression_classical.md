# Cox Proportional Hazards Regression

`cox_regression_classical` fits a federated classical Cox proportional hazards
model for time-to-event data. It estimates covariate effects with partial
likelihood and uses Breslow handling for tied event times.

## Inputs

- `y`: one positive numerical follow-up time variable.
- `x`: the event variable plus one or more covariates.
- `event_var`: the variable from `x` used to build the binary event vector.
- `positive_class`: optional event value that should be interpreted as event
  `1`.

Categorical covariates are one-hot encoded through the standard Exaflow
federated preprocessing pipeline before the Cox fit. The event variable is not
one-hot encoded; it is converted separately into a single binary event vector.

## Event Coding

The Cox model is fitted with a binary event vector. If `event_var` is stored as
`0`/`1` or `false`/`true`, leave `positive_class` empty; the algorithm detects
that coding
automatically.

Use `positive_class` when `event_var` is categorical, for example diagnosis
category or vital status. The selected level is converted to event `1`, and all
other observed levels are converted to event `0`. For example,
`positive_class="AD"` models AD as the event of interest and treats CN, MCI, and
other categories as event `0`.

The current specification schema can populate dropdowns from selected `y` values
or from a single selected `x` value. In this algorithm, `event_var` is selected
from a multi-variable `x` input, so `positive_class` cannot currently be exposed
as a dynamic dropdown of the selected event variable's CDE enumerations.

## Outputs

The algorithm returns:

- coefficient estimates
- hazard ratios
- standard errors
- Wald z-scores
- p-values
- 95% confidence intervals
- partial log-likelihood
- optimization diagnostics (`n_iter`, `converged`, `score_norm`, `step_norm`)

## Notes

- This is the exact classical Cox PH formulation, not the stacked logistic
  approximation used by `cox_regression_stacked`.
- The current implementation supports `ties="breslow"` only.
- The standalone benchmark compares the federated output against
  `statsmodels.duration.hazard_regression.PHReg`.
