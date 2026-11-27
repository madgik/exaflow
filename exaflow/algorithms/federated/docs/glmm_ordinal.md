# Ordinal generalized linear mixed model

Implementation: [mixed_effects/glmm_ordinal.py](../mixed_effects/glmm_ordinal.py).

`FederatedGLMMOrdinal(K=..., ...).fit(X, y, *, center_ids, agg_client)` fits
ordered outcomes encoded as integers `0` through `K-1`. `K >= 2`, numerical
features, aligned group ids, and a shared category order are required.

## Statistical method

```text
P(y_ij <= k | x_ij, u_j) = sigmoid(cutpoint_k - x_ij beta - u_j)
u_j ~ Normal(0, sigma_u2)
```

Random-intercept marginal likelihood uses a Laplace approximation. Positive
cutpoint gaps are parameterized exponentially to preserve ordering. With
`fit_intercept=True`, the first cutpoint is fixed at zero for identifiability;
without an intercept it is estimated.

Defaults include `max_iters=50`, `ridge=1e-6`, `tol_theta=1e-6`,
`tol_score=1e-4`, `max_step_norm=5.0`, `mode_tol=1e-8`, and
`mode_max_iter=50`. Initial random-intercept variance is `0.3`, with default
log-variance clipping corresponding to `[1e-8, 1e3]`. All configuration is
keyword-only; see the constructor for the complete interface.

### Federated flow

```text
Validate arrays, ordinal codes, category count, and solver settings.
Aggregate row counts and group identities; initialize common parameters.
Solve local group modes and compute the Laplace objective/gradient.
Aggregate objective and derivative summaries for shared parameter updates.
Preserve ordered cutpoints and use stabilized steps/line search.
Compute observed information, fixed-effect inference, and diagnostics.
```

## Results and limitations

`FederatedGLMMOrdinalResults` contains `theta`, fixed-effect `params`, `bse`,
`zvalues`, `pvalues`, confidence bounds, `cov_params`, `sigma_u2`,
`cutpoints`, `ll_laplace`, `aic`, `bic`, `nobs`, `n_groups`, `converged`,
`n_iter`, and `K`. `return_history=True` retains history. `predict_proba(X)`
returns category probabilities using fixed effects; `predict(X)` selects the
most probable category without adding group modes.

The proportional-odds model and Laplace approximation limit interpretation;
category coding/order must be consistent. Group modes are formed locally, so
partitioning must preserve the intended groups. Tests compare distributed
fitting with centralized fitting of this estimator, verify increasing
cutpoints and normalized probabilities, and compare analytic gradients with
finite differences. They do not claim equivalence with another package.
