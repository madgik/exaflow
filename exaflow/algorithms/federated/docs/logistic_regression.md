## Logistic Regression (FederatedLogisticRegression)

### Name

**Logistic Regression (FederatedLogisticRegression)**

### Type

**Statistical Model** (supervised binary classification)

### Goal (Why we need it)

Logistic Regression models the probability of a **binary outcome** using a logistic function.
In a federated setting, we want the *same* maximum likelihood estimates (coefficients, standard errors, p-values, confidence intervals) as centralized logistic regression **without sharing raw data**, using only aggregated gradient and Hessian information.

### When to use

Use Logistic Regression when:

- you need **binary classification** with probabilistic interpretation
- you want interpretable coefficients (log-odds ratios)
- you need statistical inference (p-values, confidence intervals, AIC/BIC)
- features are numerical or properly encoded categorical
- you have sufficient sample size relative to number of features

### When NOT to use

Avoid / be careful when:

- outcome is not binary (use multinomial or ordinal models)
- features are highly collinear (Hessian may be singular)
- extreme class imbalance without proper handling
- non-linear decision boundaries (consider polynomial features or other models)
- very small sample size (n < 10\*p is risky)
- complete or quasi-complete separation exists

______________________________________________________________________

### Inputs / Outputs

| Item | Description |
| -------------------- | ------------------------------------------------------- |
| **X** | Feature matrix, shape `(n_local, p)` |
| **y** | Binary target vector, shape `(n_local,)` |
| **fit_intercept** | Whether to fit intercept term (default: `True`) |
| **MAX_ITER** | Fixed maximum IRLS iterations: 50 |
| **TOL** | Fixed convergence tolerance: 1e-4 |
| **ALPHA** | Fixed confidence-interval significance level: 0.05 |
| **agg_client** | Federated aggregation client |

**Outputs (FederatedLogisticRegressionResults)**

- `params`: coefficient estimates (including intercept if fitted)
- `stderr`: standard errors of coefficients
- `z_scores`: z-statistics
- `pvalues`: two-tailed p-values
- `lower_ci`, `upper_ci`: confidence interval bounds
- `ll`: log-likelihood
- `r_squared_cs`: Cox-Snell pseudo RΒ²
- `r_squared_mcf`: McFadden pseudo RΒ²
- `aic`, `bic`: information criteria
- `predict(X)`: predicted class probabilities

### Input validation

`fit_intercept` is checked at fitting time and accepts Python or NumPy booleans.
The iteration limit, tolerance, and significance level above remain module
constants, not constructor parameters.

Features and targets retain their existing NumPy float conversion. Features
must be finite, and effective target values must be finite and encoded as `0`
or `1`. Existing target flattening, feature reshaping, intercept addition, and
row-alignment transposition are retained in their original order. With no
intercept, the existing promotion of scalar/flat training features to 2D remains
available. Validation checks matching rows after those conversions; it does not
impute values or drop observations.

Empty local partitions, a single local outcome class, and zero-feature input
with an intercept are permitted. The existing global requirements remain:
the observation count and each outcome-class count must exceed the number of
fitted parameters, including the intercept when present.

Local parameter and input validation raises `BadInputError` before estimation
without adding aggregation calls. The caller owns distributed error propagation,
cancellation, and cleanup. Exareme3 uses its existing request-error and cleanup
handling; independent callers must provide their own lifecycle handling.
Participants must still use the same configuration and aligned feature layout
for estimation. Transport/programming errors are not converted to input errors.

Prediction checks finite numeric features and compatibility with the fitted
parameter count after its existing conversions. Flat prediction input remains
a single feature column with an intercept, or one observation without an
intercept. Empty-result prediction behavior is unchanged.

Invalid inputs now fail explicitly at the package boundary rather than producing
incidental NumPy errors, invalid likelihood calculations, or non-finite results.
Global sample-count errors now advise selecting more observations or fewer
covariates. Public signatures, successful result fields, and predicted
probabilities are unchanged. Reverting the validation integration and its helper
imports restores the preceding behavior without a data or deployment migration.

### Key Differences from statsmodels

| Aspect | statsmodels | MIP Federated Implementation |
| ------------------ | --------------------- | ----------------------------- |
| Data access | Full centralized data | Data remains local per client |
| Algorithm | IRLS (Newton-Raphson) | Federated IRLS |
| Gradients/Hessian | Local computation | Federated aggregation |
| Convergence check | Max absolute gradient | Max absolute gradient |
| Performance | CPU/memory bound | Network + aggregation bound |
| Results API | Full statsmodels API | Simplified results container |

### Approximation vs Exactness

| Component | statsmodels | MIP |
| --------------------------- | ----------- | ---------------------- |
| Coefficient estimates | Exact MLE | Exact MLE |
| Standard errors | Exact | Exact |
| Log-likelihood | Exact | Exact |
| P-values | Exact | Exact |
| Confidence intervals | Exact | Exact |
| Pseudo RΒ² | Exact | Exact |
| AIC/BIC | Exact | Exact |
| Convergence detection | Exact | Exact |

______________________________________________________________________
