## 8. anova_twoway.py - FederatedAnovaTwoWay

### Name

**Two-Way ANOVA (FederatedAnovaTwoWay)**

### Type

**Statistical Test** (factorial design)

### Goal (Why we need it)

Tests for **main effects and interactions** between two grouping factors.
In a federated setting, we want the *same* F-statistics for main effects and interaction as centralized computation **without sharing raw data**, using aggregated sufficient statistics per cell.

### When to use

Use Two-Way ANOVA when:

* you have **two grouping factors**
* you want to test main effects and interaction
* the outcome variable is continuous
* groups are approximately normally distributed with equal variances

### When NOT to use

Avoid / be careful when:

* you have only one factor (use one-way ANOVA)
* you have more than two factors (use higher-order ANOVA)
* assumptions are not satisfied

---

### Inputs / Outputs

| Item           | Description                              |
| -------------- | ---------------------------------------- |
| **data**       | DataFrame with outcome and factors       |
| **value_var**  | Name of continuous outcome variable      |
| **factor1_var**| Name of first factor                     |
| **factor2_var**| Name of second factor                    |
| **agg_client** | Federated aggregation client             |

**Outputs (dict)**

* `f_stat_factor1`: F-statistic for factor 1 main effect
* `f_stat_factor2`: F-statistic for factor 2 main effect
* `f_stat_interaction`: F-statistic for interaction
* `p_value_factor1`: p-value for factor 1
* `p_value_factor2`: p-value for factor 2
* `p_value_interaction`: p-value for interaction
* `df_factor1`, `df_factor2`, `df_interaction`: degrees of freedom
* `ss_factor1`, `ss_factor2`, `ss_interaction`, `ss_error`: sums of squares

### Key Differences from statsmodels

| Aspect          | statsmodels       | MIP Federated Implementation  |
| --------------- | ----------------- | ----------------------------- |
| Data access     | Centralized       | Data remains local per client |
| F-statistics    | Exact             | Exact                         |
| P-values        | Exact             | Exact                         |

### Approximation vs Exactness

| Component       | statsmodels | MIP   |
| --------------- | ----------- | ----- |
| Cell means      | Exact       | Exact |
| SS effects      | Exact       | Exact |
| F-statistics    | Exact       | Exact |
| P-values        | Exact       | Exact |

---


