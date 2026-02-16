## 7. anova_oneway.py - FederatedAnovaOneWay

### Name

**One-Way ANOVA (FederatedAnovaOneWay)**

### Type

**Statistical Test** (multi-group comparison)

### Goal (Why we need it)

Tests for **differences in means across multiple groups** (one grouping factor).
In a federated setting, we want the *same* F-statistic and p-value as centralized computation **without sharing raw data**, using only aggregated sufficient statistics per group.

### When to use

Use One-Way ANOVA when:

* you need to compare **three or more groups**
* the outcome variable is continuous
* groups are approximately normally distributed with equal variances
* you want an overall test for group differences
* you need post-hoc tests (Tukey HSD)

### When NOT to use

Avoid / be careful when:

* you have only two groups (use t-test)
* variances are very unequal across groups
* distributions are highly non-normal
* you have multiple factors (use two-way ANOVA)

---

### Inputs / Outputs

| Item           | Description                              |
| -------------- | ---------------------------------------- |
| **groups**     | Sequence of arrays, one per group        |
| **categories** | Labels for each group                    |
| **alpha**      | Significance level for Tukey HSD         |
| **agg_client** | Federated aggregation client             |

**Outputs (dict)**

* `f_stat`: F-statistic
* `df_between`: degrees of freedom between groups
* `df_within`: degrees of freedom within groups
* `p_value`: p-value
* `ss_between`: sum of squares between groups
* `ss_within`: sum of squares within groups
* `ms_between`: mean square between
* `ms_within`: mean square within
* `tukey_results`: Tukey HSD post-hoc comparisons

### Key Differences from scipy/statsmodels

| Aspect          | scipy.stats.f_oneway | MIP Federated Implementation  |
| --------------- | -------------------- | ----------------------------- |
| Data access     | Centralized          | Data remains local per client |
| F-statistic     | Exact                | Exact                         |
| P-values        | Exact                | Exact                         |
| Post-hoc tests  | Separate function    | Integrated (Tukey HSD)        |

### Approximation vs Exactness

| Component       | scipy/statsmodels | MIP   |
| --------------- | ----------------- | ----- |
| Group means     | Exact             | Exact |
| SS between/within | Exact           | Exact |
| F-statistic     | Exact             | Exact |
| P-value         | Exact             | Exact |
| Tukey HSD       | Exact             | Exact |

---


