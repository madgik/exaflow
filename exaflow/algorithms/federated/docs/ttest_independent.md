## 4. ttest_independent.py - FederatedTTestIndependent

### Name

**Independent T-Test (FederatedTTestIndependent)**

### Type

**Statistical Test** (two-sample comparison)

### Goal (Why we need it)

Compares **means of two independent groups** under the assumption of equal (pooled) variances.
In a federated setting, we want the *same* t-statistic, p-value, and Cohen's d as centralized computation **without sharing raw data**, using only aggregated sums, sums of squares, and counts per group.

### When to use

Use Independent T-Test when:

* you need to compare **two independent groups**
* the outcome variable is continuous
* groups are approximately normally distributed (or large sample)
* variances are approximately equal (pooled variance assumption)

### When NOT to use

Avoid / be careful when:

* groups are paired/matched (use paired t-test)
* variances are very unequal (consider Welch's t-test)
* distributions are highly non-normal with small samples
* outcome is not continuous

---

### Inputs / Outputs

| Item            | Description                                  |
| --------------- | -------------------------------------------- |
| **data**        | DataFrame with grouping variable and outcome |
| **group_var**   | Name of grouping variable                    |
| **value_var**   | Name of continuous outcome variable          |
| **group_a**     | Value identifying group A                    |
| **group_b**     | Value identifying group B                    |
| **alpha**       | Significance level                           |
| **alternative** | "two-sided", "greater", or "less"            |
| **agg_client**  | Federated aggregation client                 |

**Outputs (dict)**

* `t_stat`: t-statistic
* `df`: degrees of freedom
* `p_value`: p-value for specified alternative
* `mean_diff`: difference in means (group_a - group_b)
* `se_diff`: standard error of difference
* `ci_lower`, `ci_upper`: confidence interval bounds
* `cohens_d`: effect size (Cohen's d)

### Key Differences from scipy.stats

| Aspect          | scipy.stats.ttest_ind | MIP Federated Implementation  |
| --------------- | --------------------- | ----------------------------- |
| Data access     | Centralized           | Data remains local per client |
| Variance        | Pooled or Welch       | Pooled only                   |
| T-statistic     | Exact                 | Exact                         |
| P-values        | Exact                 | Exact                         |

### Approximation vs Exactness

| Component       | scipy.stats | MIP   |
| --------------- | ----------- | ----- |
| Mean per group  | Exact       | Exact |
| SD per group    | Exact       | Exact |
| Pooled variance | Exact       | Exact |
| T-statistic     | Exact       | Exact |
| P-value         | Exact       | Exact |
| Cohen's d       | Exact       | Exact |

---


