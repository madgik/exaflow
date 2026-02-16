## Ordinary Least Squares (FederatedOLS)

### Name

**Ordinary Least Squares (FederatedOLS)**

### Type

**Statistical Model** (supervised regression)

### Goal (Why we need it)

OLS fits a **linear regression model** by minimizing squared residuals.
In a federated setting, we want the *same* regression estimates (coefficients, RΒ², F-statistic, t-tests) as centralized OLS **without sharing raw data**, using only aggregated sufficient statistics (X'X, X'y, y'y).

### When to use

Use OLS when:

* you need **continuous outcome prediction** with linear relationships
* you want interpretable coefficients
* you need statistical inference (t-tests, F-tests, RΒ², confidence intervals)
* features are numerical or properly encoded categorical
* residuals are approximately normal with constant variance

### When NOT to use

Avoid / be careful when:

* outcome is binary/categorical (use logistic/multinomial regression)
* features are highly collinear (X'X may be singular)
* heteroscedasticity is severe (consider robust standard errors)
* non-linear relationships exist (consider transformations or polynomial features)
* outliers are influential (consider robust regression)
* sample size is very small relative to features

---

### Inputs / Outputs

| Item              | Description                                          |
| ----------------- | ---------------------------------------------------- |
| **X**             | Feature matrix, shape `(n_local, p)`                 |
| **y**             | Continuous target vector, shape `(n_local,)`         |
| **fit_intercept** | Whether to fit intercept term (default: `True`)      |
| **alpha**         | Significance level for confidence intervals (0.05)   |
| **agg_client**    | Federated aggregation client                         |

**Outputs (FederatedOLSResults)**

* `params`: coefficient estimates (including intercept if fitted)
* `bse`: standard errors of coefficients
* `tvalues`: t-statistics
* `pvalues`: two-tailed p-values
* `r_squared`: coefficient of determination
* `r_squared_adjusted`: adjusted RΒ²
* `fvalue`: F-statistic
* `f_pvalue`: p-value for F-test
* `rse`: residual standard error
* `ll`, `aic`, `bic`: likelihood and information criteria
* `predict(X)`: predicted values
* `conf_int(alpha)`: confidence intervals for coefficients

### Key Differences from statsmodels

| Aspect             | statsmodels           | MIP Federated Implementation  |
| ------------------ | --------------------- | ----------------------------- |
| Data access        | Full centralized data | Data remains local per client |
| Algorithm          | Direct matrix solve   | Federated sufficient stats    |
| Sufficient stats   | Local X'X, X'y        | Aggregated X'X, X'y, y'y      |
| Performance        | CPU/memory bound      | Network + aggregation bound   |
| Results API        | Full statsmodels API  | Simplified results container  |

### Approximation vs Exactness

| Component                   | statsmodels | MIP                    |
| --------------------------- | ----------- | ---------------------- |
| Coefficient estimates       | Exact       | Exact                  |
| Standard errors             | Exact       | Exact                  |
| RΒ² / Adjusted RΒ²            | Exact       | Exact                  |
| F-statistic                 | Exact       | Exact                  |
| T-statistics / P-values     | Exact       | Exact                  |
| Confidence intervals        | Exact       | Exact                  |
| AIC/BIC                     | Exact       | Exact                  |
| Residuals                   | Local only  | Not computed globally  |

