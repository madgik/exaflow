# Linear Model Algorithms

## Table of Contents

1. [Logistic Regression](#logistic-regression-federatedlogisticregression) - Binary classification with probabilistic interpretation
2. [Ordinary Least Squares](#ordinary-least-squares-federatedols) - Linear regression

---

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

* you need **binary classification** with probabilistic interpretation
* you want interpretable coefficients (log-odds ratios)
* you need statistical inference (p-values, confidence intervals, AIC/BIC)
* features are numerical or properly encoded categorical
* you have sufficient sample size relative to number of features

### When NOT to use

Avoid / be careful when:

* outcome is not binary (use multinomial or ordinal models)
* features are highly collinear (Hessian may be singular)
* extreme class imbalance without proper handling
* non-linear decision boundaries (consider polynomial features or other models)
* very small sample size (n < 10*p is risky)
* complete or quasi-complete separation exists

---

### Inputs / Outputs

| Item                 | Description                                             |
| -------------------- | ------------------------------------------------------- |
| **X**                | Feature matrix, shape `(n_local, p)`                    |
| **y**                | Binary target vector, shape `(n_local,)`                |
| **fit_intercept**    | Whether to fit intercept term (default: `True`)         |
| **max_iter**         | Maximum IRLS iterations (default: 50)                   |
| **tol**              | Convergence tolerance (default: 1e-4)                   |
| **alpha**            | Significance level for confidence intervals (0.05)      |
| **agg_client**       | Federated aggregation client                            |

**Outputs (FederatedLogisticRegressionResults)**

* `params`: coefficient estimates (including intercept if fitted)
* `stderr`: standard errors of coefficients
* `z_scores`: z-statistics
* `pvalues`: two-tailed p-values
* `lower_ci`, `upper_ci`: confidence interval bounds
* `ll`: log-likelihood
* `r_squared_cs`: Cox-Snell pseudo R²
* `r_squared_mcf`: McFadden pseudo R²
* `aic`, `bic`: information criteria
* `predict(X)`: predicted class probabilities

### Key Differences from statsmodels

| Aspect             | statsmodels           | MIP Federated Implementation  |
| ------------------ | --------------------- | ----------------------------- |
| Data access        | Full centralized data | Data remains local per client |
| Algorithm          | IRLS (Newton-Raphson) | Federated IRLS                |
| Gradients/Hessian  | Local computation     | Federated aggregation         |
| Convergence check  | Max absolute gradient | Max absolute gradient         |
| Performance        | CPU/memory bound      | Network + aggregation bound   |
| Results API        | Full statsmodels API  | Simplified results container  |

### Approximation vs Exactness

| Component                   | statsmodels | MIP                    |
| --------------------------- | ----------- | ---------------------- |
| Coefficient estimates       | Exact MLE   | Exact MLE              |
| Standard errors             | Exact       | Exact                  |
| Log-likelihood              | Exact       | Exact                  |
| P-values                    | Exact       | Exact                  |
| Confidence intervals        | Exact       | Exact                  |
| Pseudo R²                   | Exact       | Exact                  |
| AIC/BIC                     | Exact       | Exact                  |
| Convergence detection       | Exact       | Exact                  |

---

## Ordinary Least Squares (FederatedOLS)

### Name

**Ordinary Least Squares (FederatedOLS)**

### Type

**Statistical Model** (supervised regression)

### Goal (Why we need it)

OLS fits a **linear regression model** by minimizing squared residuals.
In a federated setting, we want the *same* regression estimates (coefficients, R², F-statistic, t-tests) as centralized OLS **without sharing raw data**, using only aggregated sufficient statistics (X'X, X'y, y'y).

### When to use

Use OLS when:

* you need **continuous outcome prediction** with linear relationships
* you want interpretable coefficients
* you need statistical inference (t-tests, F-tests, R², confidence intervals)
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
* `r_squared_adjusted`: adjusted R²
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
| R² / Adjusted R²            | Exact       | Exact                  |
| F-statistic                 | Exact       | Exact                  |
| T-statistics / P-values     | Exact       | Exact                  |
| Confidence intervals        | Exact       | Exact                  |
| AIC/BIC                     | Exact       | Exact                  |
| Residuals                   | Local only  | Not computed globally  |
