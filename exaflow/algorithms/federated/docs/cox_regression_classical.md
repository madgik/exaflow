# Classical Cox regression

Implementation: [linear_model/cox_regression_classical.py](../linear_model/cox_regression_classical.py).

Cox proportional hazards regression models time-to-event data using the partial
likelihood. It estimates covariate effects on the hazard while leaving the
baseline hazard unspecified.

## Library interface

`FederatedClassicalCoxRegression(ties="breslow", tol=1e-6, max_iter=100, ridge=1e-8, max_step_norm=5.0, verbose=False).fit(X, y, *, agg_client, feature_names=None)` takes numeric features and a two-column target: follow-up time and binary event indicator. Constructor settings are keyword-only. The result exposes covariate estimates, hazard ratios, uncertainty, likelihood, and convergence information.

## Statistical method

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
Receive numeric X and y[:, 0] = time, y[:, 1] = binary event.
Validate strictly positive times and discover the union of event times.
Aggregate event counts and event covariate sums on this grid.
Repeat Newton updates:
    compute local risk-set S0, S1, and S2 at current beta
    sum risk-set quantities across participants
    compute Breslow likelihood, score, and information
    take a stabilized Newton step and check convergence
Return coefficients, hazard ratios, uncertainty, and diagnostics.
```

## Numerical decisions

- Breslow handling is used for tied event times.
- Time values must be strictly positive.
- Exponentiation is clipped for numerical stability.
- Newton updates use regularization and a pseudo-inverse fallback when needed.

## Reference comparison

Standalone Cox tests use `statsmodels.duration.hazard_regression.PHReg` as the
centralized reference for classical Cox proportional hazards behavior.

Reference behavior is aligned with Cox partial-likelihood estimation using
Breslow tie handling.

## Limitations

- Follow-up times must be strictly positive.
- At least one observed event is required.
- The proportional hazards assumption is not tested automatically.
- Time-varying covariates are not supported.
- Event indicators supplied to the estimator must already be binary.
- High collinearity can make the information matrix unstable.
