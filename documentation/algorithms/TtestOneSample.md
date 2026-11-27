# One-sample t-test

## Overview

The one-sample t-test compares the mean of a numerical variable with a reference
mean `mu`. It returns the t statistic, p-value, confidence interval, standard
error, and Cohen's d.

## Inputs

### Required inputs

| Input | Description |
|---|---|
| `y` | Numerical variable to test. |

### Parameters

| Parameter | Description | Default |
|---|---|---|
| `mu` | Mean value under the null hypothesis. | `0.0` |
| `alpha` | Significance level for confidence intervals. | `0.05` |
| `alt_hypothesis` | Alternative hypothesis: `two-sided`, `less`, or `greater`. | `two-sided` |

## Method

The mathematical method, library API, aggregation flow, numerical decisions, reference comparisons, and limitations are documented in [One-sample t-test](../../exaflow/algorithms/federated/docs/ttest_one_sample.md).

## Outputs

| Field | Description |
|---|---|
| `n_obs` | Number of observations used. |
| `std` | Sample standard deviation. |
| `t_stat` | t statistic. |
| `df` | Degrees of freedom. |
| `p` | P-value. |
| `mean_diff` | Sample mean. |
| `se_diff` | Standard error of the mean. |
| `ci_lower` | Lower confidence interval bound. |
| `ci_upper` | Upper confidence interval bound. |
| `cohens_d` | Standardized mean difference. |
