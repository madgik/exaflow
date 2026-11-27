# Independent t-test

## Overview

The independent t-test compares the mean of a numerical variable between two
independent groups. This implementation uses the pooled-variance Student t-test
and reports the t statistic, p-value, confidence interval, standard error, and
Cohen's d.

## Inputs

### Required inputs

| Input | Description |
|---|---|
| `y` | Numerical outcome variable. |
| `x` | Categorical grouping variable. |

### Parameters

| Parameter | Description | Default |
|---|---|---|
| `groupA` | First group category. | Required |
| `groupB` | Second group category. | Required |
| `alpha` | Significance level for confidence intervals. | `0.05` |
| `alt_hypothesis` | Alternative hypothesis: `two-sided`, `less`, or `greater`. | `two-sided` |

## Method

The mathematical method, library API, aggregation flow, numerical decisions, reference comparisons, and limitations are documented in [Independent-samples t-test](../../exaflow/algorithms/federated/docs/ttest_independent.md).

## Outputs

| Field | Description |
|---|---|
| `t_stat` | t statistic. |
| `df` | Degrees of freedom. |
| `p` | P-value. |
| `mean_diff` | `mean(groupA) - mean(groupB)`. |
| `se_diff` | Standard error of the mean difference. |
| `ci_lower` | Lower confidence interval bound. |
| `ci_upper` | Upper confidence interval bound. |
| `cohens_d` | Standardized mean difference using pooled standard deviation. |
