## 3. pearson_correlation.py - FederatedPearsonCorrelation

### Name

**Pearson Correlation (FederatedPearsonCorrelation)**

### Type

**Statistical Test** (correlation analysis)

### Goal (Why we need it)

Computes **Pearson correlation** between pairs of continuous variables with correlation coefficients, p-values, and confidence intervals.
In a federated setting, we want the *same* correlations and inference as centralized computation **without sharing raw data**, using only aggregated sums and cross-products.

### When to use

Use Pearson Correlation when:

* you need to measure **linear associations** between continuous variables
* you want statistical inference (p-values, confidence intervals)
* variables are approximately normally distributed (or large sample)
* you need to identify collinear features

### When NOT to use

Avoid / be careful when:

* relationships are non-linear (consider Spearman rank correlation)
* variables have severe outliers
* variables are not continuous
* sample size is very small

---

### Inputs / Outputs

| Item           | Description                                |
| -------------- | ------------------------------------------ |
| **data**       | DataFrame with numerical features          |
| **x_vars**     | List of x-variable names                   |
| **y_vars**     | List of y-variable names                   |
| **alpha**      | Significance level for confidence intervals|
| **agg_client** | Federated aggregation client               |

**Outputs (PearsonCorrelationResult)**

* `correlations`: correlation matrix (x_vars Γ— y_vars)
* `p_values`: p-value matrix
* `ci_lo`, `ci_hi`: confidence interval bounds (Fisher z-transform)
* `n_obs`: total observations

### Key Differences from scipy/statsmodels

| Aspect          | scipy.stats     | MIP Federated Implementation  |
| --------------- | --------------- | ----------------------------- |
| Data access     | Centralized     | Data remains local per client |
| Correlation     | Exact           | Exact                         |
| P-values        | Exact           | Exact                         |
| Conf. intervals | Exact           | Exact                         |

### Approximation vs Exactness

| Component            | scipy/statsmodels | MIP   |
| -------------------- | ----------------- | ----- |
| Correlation coef     | Exact             | Exact |
| P-values (t-test)    | Exact             | Exact |
| Confidence intervals | Exact             | Exact |

---


