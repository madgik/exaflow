# Statistical Algorithms

This directory contains federated implementations for statistical analyses and descriptive statistics.

## Table of Contents

1. [FederatedDescribe](#1-describepy---federateddescribe) - Descriptive statistics for numerical and categorical variables
2. [FederatedDescriptiveStatistics](#2-descriptive_statspy---federateddescriptivestatistics) - Unified wrapper interface
3. [FederatedPearsonCorrelation](#3-pearson_correlationpy---federatedpearsoncorrelation) - Pearson correlation analysis
4. [FederatedTTestIndependent](#4-ttest_independentpy---federatedttestindependent) - Independent samples t-test
5. [FederatedTTestOneSample](#5-ttest_onesamplepy---federatedttestonesample) - One-sample t-test
6. [FederatedTTestPaired](#6-ttest_pairedpy---federatedttestpaired) - Paired samples t-test
7. [FederatedAnovaOneWay](#7-anova_onewaypy---federatedanovaoneway) - One-way ANOVA
8. [FederatedAnovaTwoWay](#8-anova_twowaypy---federatedanovatwoway) - Two-way ANOVA
9. [FederatedHistogram](#9-histogrampy---federatedhistogram) - Histogram computation

---

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

* you need **exploratory data analysis** to understand data distributions
* you want summary statistics for numerical variables (mean, std, quantiles)
* you want frequency counts for categorical variables
* you need to check data quality (missing values, outliers)
* you want statistics grouped by dataset (per-site summaries + global)

### When NOT to use

* When you only need one specific statistic (use specialized function)
* For streaming data with changing distributions

---

### Inputs / Outputs

| Item                | Description                                               |
| ------------------- | --------------------------------------------------------- |
| **data**            | DataFrame with numerical and categorical features         |
| **numerical_vars**  | List of numerical variable names                          |
| **nominal_vars**    | List of categorical variable names                        |
| **nominal_levels**  | Dict mapping categorical variables to category lists      |
| **min_row_count**   | Minimum rows required per dataset                         |
| **dataset_col**     | Column name for dataset/site identification               |
| **agg_client**      | Federated aggregation client                              |

**Outputs (DescribeResult)**

* `recs_varbased`: list of per-variable stats per dataset
* `recs_modbased`: list of per-modality (category) stats per dataset
* `global_varbased`: list of per-variable global aggregated stats
* `global_modbased`: list of per-modality global aggregated stats

Each record includes: count, mean, std, min, q25, q50 (median), q75, max, mode, missing_count

### Key Differences from pandas/statsmodels

| Aspect               | pandas describe       | MIP Federated Implementation  |
| -------------------- | --------------------- | ----------------------------- |
| Data access          | Full centralized data | Data remains local per client |
| Median/Quantiles     | Exact (sorting)       | Approximate (histogram-based) |
| Mode                 | Exact                 | Approximate (top frequencies) |
| Min/Max              | Exact                 | Exact                         |
| Mean/Std             | Exact                 | Exact                         |
| Categorical freqs    | Exact                 | Exact                         |

### Approximation vs Exactness

| Component          | pandas/statsmodels | MIP                      |
| ------------------ | ------------------ | ------------------------ |
| Count              | Exact              | Exact                    |
| Mean               | Exact              | Exact                    |
| Std                | Exact              | Exact                    |
| Min / Max          | Exact              | Exact                    |
| Median / Quantiles | Exact              | Approximate (histograms) |
| Mode               | Exact              | Approximate (top-k)      |
| Frequencies        | Exact              | Exact                    |

---

## 2. descriptive_stats.py - FederatedDescriptiveStatistics

### Name

**Descriptive Statistics Wrapper (FederatedDescriptiveStatistics)**

### Type

**Convenience Wrapper** (unified interface)

### Goal (Why we need it)

Provides a **unified interface** that combines describe, histogram and correlation functions into a statsmodels-style API.
It is a wrapper class that calls the underlying algorithms (FederatedDescribe, FederatedHistogram, FederatedPearsonCorrelation).

### When to use

Use FederatedDescriptiveStatistics when:

* you want a **unified API** for multiple descriptive statistics
* you prefer a statsmodels-style interface
* you need quick access to describe, hist, corrcoef through one object

### When NOT to use

* When you only need one of the underlying functions (use it directly)

---

### Inputs / Outputs

**Available Methods:**

* `describe()`: Calls FederatedDescribe
* `hist()`: Calls FederatedHistogram
* `corrcoef()`: Calls FederatedPearsonCorrelation
* `pearson_correlation()`: Alias for corrcoef()

Inputs/Outputs are the same as the underlying algorithms.

### Key Differences from statsmodels

| Aspect      | statsmodels           | MIP Implementation         |
| ----------- | --------------------- | -------------------------- |
| Purpose     | Full statistical API  | Convenience wrapper only   |
| Scope       | Broader functionality | Descriptive stats focused  |

### Approximation vs Exactness

Inherits exactness/approximation from the underlying algorithms.

---

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

* `correlations`: correlation matrix (x_vars × y_vars)
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

## 9. histogram.py - FederatedHistogram

### Name

**Histogram (FederatedHistogram)**

### Type

**Descriptive Statistics** (distribution visualization)

### Goal (Why we need it)

Computes **binned frequency counts** for visualizing data distributions.
In a federated setting, bins are defined globally and each client computes local counts that are aggregated.

### When to use

Use Histogram when:

* you need to visualize **continuous variable distributions**
* you want to identify skewness, modes, outliers
* binning strategy is known or can be defined
* supports both numerical and categorical histograms

### When NOT to use

Avoid / be careful when:

* data are categorical without predefined levels (use frequency tables)
* bin edges are unknown and cannot be pre-specified

---

### Inputs / Outputs

| Item              | Description                                   |
| ----------------- | --------------------------------------------- |
| **data**          | DataFrame with variable                       |
| **y_var**         | Variable for histogram                        |
| **x_vars**        | Optional grouping variables                   |
| **metadata**      | Metadata with enumerations or bin info        |
| **bins**          | Number of bins (for numerical)                |
| **min_row_count** | Minimum count for privacy masking             |
| **agg_client**    | Federated aggregation client                  |

**Outputs (HistogramResult)**

* `bins`: list of bin edges or category labels
* `counts`: list of frequencies per bin
* `grouped`: dict of histograms per grouping variable

### Key Differences from numpy/matplotlib

| Aspect          | numpy.histogram   | MIP Federated Implementation  |
| --------------- | ----------------- | ----------------------------- |
| Data access     | Centralized       | Data remains local per client |
| Bin computation | Local             | Requires pre-specified bins   |
| Counts          | Exact             | Exact (aggregated)            |
| Privacy masking | None              | min_row_count support         |

### Approximation vs Exactness

| Component       | numpy/matplotlib | MIP                   |
| --------------- | ---------------- | --------------------- |
| Bin edges       | Exact            | Pre-specified         |
| Counts per bin  | Exact            | Exact (aggregated)    |
| Grouping        | N/A              | Exact (aggregated)    |
