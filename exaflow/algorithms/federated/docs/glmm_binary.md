# Binary generalized linear mixed model

Implementation: [mixed_effects/glmm_binary.py](../mixed_effects/glmm_binary.py).

`FederatedGLMMBinary(...).fit(X, y, *, center_ids, agg_client, w=None)` accepts
numeric features, binary outcomes encoded as 0/1, local group ids, and optional
row weights. The default includes an intercept.

## Statistical method

```text
P(y_ij = 1 | x_ij, u_j) = sigmoid(x_ij beta + u_j)
u_j ~ Normal(0, sigma_u2)
```

The method approximates the random-intercept marginal likelihood using Laplace
corrections. These corrections are required; disabling them is rejected.
Each group mode is solved locally and contributes score/information summaries
to the shared Newton update.

Defaults include `max_iters=50`, `ridge=1e-6`, `tol_theta=1e-6`,
`tol_score=1e-4`, `max_step_norm=5.0`, `mode_tol=1e-8`, and
`mode_max_iter=25`. Initial random-intercept variance is `0.3`; default
log-variance clipping corresponds to variances from `1e-8` to `1e3`.
The constructor documents the remaining keyword-only options.

### Federated flow

```text
Validate arrays, encoded targets, group ids, weights, and solver settings.
Aggregate row counts and group identities; initialize shared parameters.
For each iteration:
    solve local group modes at the current fixed effects and variance
    compute Laplace-corrected score/information contributions
    aggregate summaries and take a stabilized, bounded Newton step
    check parameter and score convergence
Compute covariance, Wald inference, approximate likelihood, and diagnostics.
```

## Results and limitations

`FederatedGLMMBinaryResults` contains `theta`, fixed-effect `params`, `bse`,
`zvalues`, `pvalues`, `conf_int_low`, `conf_int_high`, `cov_params`,
`sigma_u2`, `ll_laplace`, `aic`, `bic`, `nobs`, `n_groups`, `converged`,
and `n_iter`. `return_history=True` retains iteration history. `predict(X)`
uses fixed effects to return probabilities; it does not integrate over the
random-effect distribution or add fitted group modes.

Laplace integration and solver stabilization make this an approximate model.
Small groups, separation, and boundary variances can challenge fitting. Group
modes are formed locally, so group partitioning must match the intended model.
Tests compare distributed fitting with the same estimator on centralized
partitions and check prediction/inference invariants; this does not establish
identity with another mixed-model package.
