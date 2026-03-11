## Linear Mixed Model (FederatedLMM)

### Name

**Linear Mixed Model (FederatedLMM)**

### Type

**Statistical Model** (continuous outcome, random-intercept mixed model with REML)

### Goal (Why we need it)

Estimate fixed effects and variance components (`sigma2`, `sigma_u2`) for clustered continuous outcomes while keeping row-level data local to each worker.

### When to use

Use LMM when:

* outcome is continuous
* observations are grouped (e.g., center/hospital/site)
* you need fixed-effect inference and cluster variance estimation
* random-intercept structure is clinically/statistically reasonable

### When NOT to use

Avoid / be careful when:

* outcome is binary/ordinal (use GLMM variants)
* cluster structure is not present
* you need random slopes (current implementation is random intercept only)
* cluster count is extremely small (variance estimates may be unstable)

---

### Inputs / Outputs

| Item              | Description                                          |
| ----------------- | ---------------------------------------------------- |
| **X**             | Local feature matrix, shape `(n_local, p)`           |
| **y**             | Local continuous target, shape `(n_local,)`          |
| **center_ids**    | Local cluster ids, shape `(n_local,)`                |
| **fit_intercept** | Whether to add intercept term (default: `True`)      |
| **max_iter**      | Maximum REML iterations (default: `80`)              |
| **tol**           | Convergence tolerance (default: `1e-8`)              |
| **agg_client**    | Federated aggregation client                         |
| **w**             | Optional local non-negative weights                  |

**Outputs (FederatedLMMResults)**

* `params`: fixed-effect coefficients
* `bse`, `tvalues`, `pvalues`: inference for fixed effects
* `conf_int_low`, `conf_int_high`: confidence interval bounds
* `sigma2`: residual variance
* `sigma_u2`: random-intercept variance
* `cov_params`: covariance matrix of fixed effects
* `ll_reml`, `aic`, `bic`: fit quality metrics
* `nobs`, `n_groups`, `df_model`, `df_resid`
* `converged`, `n_iter`
* `predict(X)`: predicted mean outcome

### Key Differences from statsmodels

| Aspect               | statsmodels MixedLM      | Exaflow FederatedLMM         |
| -------------------- | ------------------------ | ---------------------------- |
| Data access          | Centralized              | Stays local per worker       |
| Optimization target  | REML                     | REML                         |
| Random effects       | Broad support            | Random intercept             |
| Computation          | Single-process           | Federated aggregated updates |
| API surface          | Full statsmodels object  | Simplified results container |

### Approximation vs Exactness

| Component                    | Status in FederatedLMM               |
| --------------------------- | ------------------------------------ |
| Fixed effects               | Exact for aggregated model equations |
| Variance components         | Iterative REML optimization          |
| Inference stats             | Available for fixed effects          |
| Random slopes               | Not supported                        |
