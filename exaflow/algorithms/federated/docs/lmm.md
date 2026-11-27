# Linear mixed model

Implementation: [mixed_effects/lmm.py](../mixed_effects/lmm.py).

`FederatedLMM(...).fit(X, y, *, center_ids, agg_client, w=None)` fits a
Gaussian random-intercept model. `X` is numerical, `y` is continuous, and
`center_ids` identifies local groups. Optional weights must align with rows.
The default includes an intercept (`fit_intercept=True`).

## Statistical method

```text
y_ij = x_ij beta + u_j + epsilon_ij
u_j ~ Normal(0, sigma_u2)
epsilon_ij ~ Normal(0, sigma2)
```

The estimator profiles fixed effects through generalized least squares and
estimates variance components using a restricted-likelihood objective.
Defaults include `max_iter=80`, `min_iter=10`, `tol=1e-8`, `ridge=1e-8`,
variance bounds `[1e-6, 1e6]`, and `reg_lambda=0.01`. Rough variance
initialization is enabled; `init_sigma2=1.0` and `init_sigma_u2=0.5` are
available when explicit initialization is used. See the constructor for the
complete keyword-only configuration.

### Federated flow

```text
Validate numeric features/outcome, group ids, weights, and model settings.
Prepare group cross-products, sums, and group-size information locally.
Aggregate counts, group-size summaries, and packed sufficient statistics.
Profile beta and optimize the variance components within their bounds.
Compute fixed-effect covariance, inference, and fit diagnostics.
Return the results object without transferring observation-level data.
```

## Results and limitations

`FederatedLMMResults` exposes `params`, `bse`, `tvalues`, `pvalues`,
`conf_int_low`, `conf_int_high`, `cov_params`, `sigma2`, `sigma_u2`,
`nobs`, `n_groups`, degrees of freedom, `ll_reml`, `aic`, `bic`,
`converged`, and `n_iter`. `return_history=True` retains optimization history.
`predict(X)` returns fixed-effect predictions, without group-specific random
intercept predictions.

Group statistics are formed locally; partitioning and group identities must
represent the intended random-effect groups. Ridge stabilization, variance
bounds, and regularization can affect estimates near a boundary. This is a
random-intercept model, not a general random-slope formula interface.
Focused tests compare one- and multiple-participant results with centralized
fitting and `statsmodels.MixedLM`, including boundary-variance cases.
