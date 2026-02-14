## 5. ttest_onesample.py - FederatedTTestOneSample

### Name

**One-Sample T-Test (FederatedTTestOneSample)**

### Type

**Statistical Test** (one-sample location test)

### Goal (Why we need it)

Compares the **mean of a single sample** to a specified value (null hypothesis mean).
In a federated setting, we want the *same* t-statistic and p-value as centralized computation **without sharing raw data**, using only aggregated sum, sum of squares, and count.

### When to use

Use One-Sample T-Test when:

* you need to test if the sample mean differs from a hypothesized value
* the outcome variable is continuous
* data are approximately normally distributed (or large sample)

### When NOT to use

Avoid / be careful when:

* comparing two groups (use independent or paired t-test)
* distribution is highly non-normal with small sample
* outcome is not continuous

---

### Inputs / Outputs

| Item           | Description                            |
| -------------- | -------------------------------------- |
| **data**       | DataFrame with outcome variable        |
| **value_var**  | Name of continuous outcome variable    |
| **popmean**    | Hypothesized population mean           |
| **alpha**      | Significance level                     |
| **alternative**| "two-sided", "greater", or "less"      |
| **agg_client** | Federated aggregation client           |

**Outputs (dict)**

* `t_stat`: t-statistic
* `df`: degrees of freedom
* `p_value`: p-value
* `mean`: sample mean
* `ci_lower`, `ci_upper`: confidence interval bounds

### Key Differences from scipy.stats

| Aspect      | scipy.stats.ttest_1samp | MIP Federated Implementation  |
| ----------- | ----------------------- | ----------------------------- |
| Data access | Centralized             | Data remains local per client |
| T-statistic | Exact                   | Exact                         |
| P-values    | Exact                   | Exact                         |

### Approximation vs Exactness

| Component   | scipy.stats | MIP   |
| ----------- | ----------- | ----- |
| Mean        | Exact       | Exact |
| Std         | Exact       | Exact |
| T-statistic | Exact       | Exact |
| P-value     | Exact       | Exact |

---


