# Cox Regression Classical

`cox_regression_classical` fits a federated classical Cox proportional hazards
model using partial likelihood with Breslow treatment of tied event times.

## Inputs

- `y`: one positive numerical follow-up time variable
- `x`: one event indicator variable plus one or more covariates
- `event_var`: the variable from `x` used as the binary event indicator
- `positive_class`: optional label that should be interpreted as event `1`

Categorical covariates are one-hot encoded through the standard Exaflow
federated preprocessing pipeline before the Cox fit.

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
