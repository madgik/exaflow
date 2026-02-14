## 6. ttest_paired.py - FederatedTTestPaired

### Name

**Paired T-Test (FederatedTTestPaired)**

### Type

**Statistical Test** (paired comparison)

### Goal (Why we need it)

Compares **means of two related/matched groups** (e.g., before/after measurements).
In a federated setting, we want the *same* t-statistic and p-value as centralized computation **without sharing raw data**, using only aggregated statistics on the differences.

### When to use

Use Paired T-Test when:

* you have **matched pairs** or repeated measurements
* the outcome variable is continuous
* differences are approximately normally distributed

### When NOT to use

Avoid / be careful when:

* groups are independent (use independent t-test)
* pairs are not properly matched
* differences are highly non-normal with small sample

---

### Inputs / Outputs

| Item            | Description                            |
| --------------- | -------------------------------------- |
| **data**        | DataFrame with paired measurements     |
| **value1_var**  | Name of first variable                 |
| **value2_var**  | Name of second variable                |
| **alpha**       | Significance level                     |
| **alternative** | "two-sided", "greater", or "less"      |
| **agg_client**  | Federated aggregation client           |

**Outputs (dict)**

* `t_stat`: t-statistic
* `df`: degrees of freedom
* `p_value`: p-value
* `mean_diff`: mean difference
* `ci_lower`, `ci_upper`: confidence interval bounds

### Key Differences from scipy.stats

| Aspect      | scipy.stats.ttest_rel | MIP Federated Implementation  |
| ----------- | --------------------- | ----------------------------- |
| Data access | Centralized           | Data remains local per client |
| T-statistic | Exact                 | Exact                         |
| P-values    | Exact                 | Exact                         |

### Approximation vs Exactness

| Component        | scipy.stats | MIP   |
| ---------------- | ----------- | ----- |
| Mean difference  | Exact       | Exact |
| Std of diff      | Exact       | Exact |
| T-statistic      | Exact       | Exact |
| P-value          | Exact       | Exact |

---


