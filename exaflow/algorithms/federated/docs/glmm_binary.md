## Binary GLMM (FederatedGLMMBinary)

### Name

**Binary GLMM (FederatedGLMMBinary)**

### Type

**Statistical Model** (binary outcome, random-intercept GLMM with Laplace approximation)

### Goal (Why we need it)

Estimate fixed effects and random-intercept variance for clustered binary outcomes in a federated setup, without exchanging row-level patient data.

### When to use

Use Binary GLMM when:

* outcome is binary (`0/1`)
* data are clustered by center/site
* logistic regression without random effects is too optimistic
* you need center-level heterogeneity modeling (`sigma_u2`)

### When NOT to use

Avoid / be careful when:

* outcome is ordinal (use `FederatedGLMMOrdinal`)
* no meaningful clustering exists
* you need random slopes (not currently supported)
* events are extremely rare with tiny per-center samples

---

### Inputs / Outputs

| Item              | Description                                          |
| ----------------- | ---------------------------------------------------- |
| **X**             | Local feature matrix, shape `(n_local, p)`           |
| **y**             | Local binary labels in `{0, 1}`                      |
| **center_ids**    | Local cluster ids, shape `(n_local,)`                |
| **fit_intercept** | Whether to add intercept term (default: `True`)      |
| **max_iters**     | Maximum optimization iterations                       |
| **tol_theta**     | Parameter-step tolerance                             |
| **tol_score**     | Score-norm tolerance                                 |
| **agg_client**    | Federated aggregation client                         |

**Outputs (FederatedGLMMBinaryResults)**

* `theta`: full vector `[beta..., log_sigma_u2]`
* `params`: fixed effects `beta`
* `sigma_u2`: random-intercept variance
* `nobs`, `n_groups`
* `converged`, `n_iter`
* `predict(X)`: class probabilities
* `history` (optional): optimization diagnostics

### Key Differences from centralized GLMM

| Aspect               | Centralized GLMM         | Exaflow FederatedGLMMBinary |
| -------------------- | ------------------------ | --------------------------- |
| Data access          | Full row-level data      | Local-only + aggregated stats |
| Likelihood handling  | Direct centralized solve | Federated Newton updates with Laplace terms |
| Random effects       | Broad options            | Random intercept             |
| Compute topology     | Single node              | Multi-worker federation      |

### Approximation vs Exactness

| Component                    | Status in FederatedGLMMBinary     |
| --------------------------- | --------------------------------- |
| Fixed effects               | Iterative estimate                |
| Random-intercept variance   | Iterative estimate (`sigma_u2`)   |
| Likelihood treatment        | Laplace approximation             |
| Random slopes               | Not supported                     |
