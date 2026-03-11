## Ordinal GLMM (FederatedGLMMOrdinal)

### Name

**Ordinal GLMM (FederatedGLMMOrdinal)**

### Type

**Statistical Model** (ordered categorical outcome, random-intercept GLMM with Laplace approximation)

### Goal (Why we need it)

Model ordered outcomes (e.g., severity stages) with fixed effects, ordered cutpoints, and center-level random intercepts in a federated environment.

### When to use

Use Ordinal GLMM when:

* outcome has natural order and `K >= 2` categories
* data are clustered by center/site
* proportional-odds style cumulative logit is appropriate
* you need both fixed effects and random-intercept variability

### When NOT to use

Avoid / be careful when:

* outcome is binary only (use `FederatedGLMMBinary`)
* outcome is continuous (use `FederatedLMM`/`FederatedOLS`)
* category order is not meaningful
* you need random slopes (not currently supported)

---

### Inputs / Outputs

| Item              | Description                                          |
| ----------------- | ---------------------------------------------------- |
| **X**             | Local feature matrix, shape `(n_local, p)`           |
| **y**             | Local labels in `{0, ..., K-1}`                      |
| **center_ids**    | Local cluster ids, shape `(n_local,)`                |
| **K**             | Number of ordered categories                          |
| **fit_intercept** | Whether to add intercept term (default: `True`)      |
| **max_iters**     | Maximum optimization iterations                       |
| **tol_theta**     | Parameter-step tolerance                             |
| **tol_score**     | Score-norm tolerance                                 |
| **agg_client**    | Federated aggregation client                         |

**Outputs (FederatedGLMMOrdinalResults)**

* `theta`: full parameter vector
* `params`: fixed effects `beta`
* `cutpoints`: ordered category thresholds
* `sigma_u2`: random-intercept variance
* `nobs`, `n_groups`
* `converged`, `n_iter`
* `predict(X)`: predicted class label
* `predict_proba(X)`: class probabilities
* `history` (optional): optimization diagnostics

### Key Differences from centralized ordinal mixed models

| Aspect               | Centralized model        | Exaflow FederatedGLMMOrdinal |
| -------------------- | ------------------------ | ---------------------------- |
| Data access          | Full row-level data      | Local-only + aggregated terms |
| Ordinal structure    | Cumulative logit         | Cumulative logit             |
| Random effects       | Broad options            | Random intercept             |
| Compute topology     | Single node              | Multi-worker federation      |

### Approximation vs Exactness

| Component                    | Status in FederatedGLMMOrdinal   |
| --------------------------- | -------------------------------- |
| Fixed effects               | Iterative estimate               |
| Cutpoints                   | Iterative estimate (ordered)     |
| Random-intercept variance   | Iterative estimate (`sigma_u2`)  |
| Likelihood treatment        | Laplace approximation            |
| Random slopes               | Not supported                    |
