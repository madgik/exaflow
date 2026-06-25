## 1. describe.py - FederatedDescribe

### Name

**Descriptive Statistics (FederatedDescribe)**

### Type

**Descriptive Statistics** (exploratory data analysis)

### Goal (Why we need it)

Computes **descriptive statistics** (mean, std, min, max, quantiles, mode) for numerical and categorical variables.
In a federated setting, we want the *same* descriptive statistics as centralized computation **without sharing raw data**, using only aggregated sufficient statistics (sums, min/max, counts, frequencies).

### When to use

Use Describe when:

- you need **exploratory data analysis** to understand data distributions
- you want summary statistics for numerical variables (mean, std, quantiles)
- you want frequency counts for categorical variables
- you need to check data quality (missing values, outliers)
- you want statistics grouped by dataset (per-site summaries + global)

### When NOT to use

- When you only need one specific statistic (use specialized function)
- For streaming data with changing distributions

______________________________________________________________________

### Inputs / Outputs

| Item | Description |
| ------------------- | --------------------------------------------------------- |
| **data** | DataFrame with numerical and categorical features |
| **numerical_vars** | List of numerical variable names |
| **nominal_vars** | List of categorical variable names |
| **nominal_levels** | Dict mapping categorical variables to category lists |
| **min_row_count** | Minimum rows required per dataset |
| **dataset_col** | Column name for dataset/site identification |
| **agg_client** | Federated aggregation client |

**Outputs (DescribeResult)**

- `recs`: per-`(variable, dataset)` summary records
- `global_recs`: per-variable combined (`all datasets`) summary records

Per-dataset numerical records include: `num_dtps`, `num_na`, `num_total`,
`mean`, `std`, `min`, `q1`, `q2` (local median), `q3`, `max`. The combined
records carry the aggregated `mean`, `std`, `min`, `max` plus federated
`q1`/`q2`/`q3` and `median` (= `q2`) estimated across all datasets.

### Key Differences from pandas/statsmodels

| Aspect | pandas describe | MIP Federated Implementation |
| -------------------- | --------------------- | ----------------------------- |
| Data access | Full centralized data | Data remains local per client |
| Median/Quantiles | Exact (sorting) | Per-dataset quartiles exact (pandas); combined median is a federated histogram estimate |
| Mode | Exact | Approximate (top frequencies) |
| Min/Max | Exact | Exact |
| Mean/Std | Exact | Exact |
| Categorical freqs | Exact | Exact |

### Approximation vs Exactness

| Component | pandas/statsmodels | MIP |
| ------------------ | ------------------ | ------------------------ |
| Count | Exact | Exact |
| Mean | Exact | Exact |
| Std | Exact | Exact |
| Min / Max | Exact | Exact |
| Dataset quartiles | Exact | Exact (pandas, per dataset) |
| Combined median | Exact | Approximate (federated histogram) |
| Mode | Exact | Approximate (top-k) |
| Frequencies | Exact | Exact |

______________________________________________________________________
