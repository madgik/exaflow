# Paired t-test

## Overview

The paired t-test compares two related numerical measurements by testing whether
the mean paired difference is zero. It returns the t statistic, p-value,
confidence interval, standard error, and Cohen's d for the paired differences.

## Inputs

### Required inputs

| Input | Description |
|---|---|
| `x` | First numerical measurement. |
| `y` | Second numerical measurement. |

### Parameters

| Parameter | Description | Default |
|---|---|---|
| `alpha` | Significance level for confidence intervals. | `0.05` |
| `alt_hypothesis` | Alternative hypothesis: `two-sided`, `less`, or `greater`. | `two-sided` |

## Method

The mathematical method, library API, aggregation flow, numerical decisions, reference comparisons, and limitations are documented in [Paired-samples t-test](../../exaflow/algorithms/federated/docs/ttest_paired.md).

## Outputs

| Field | Description |
|---|---|
| `t_stat` | t statistic. |
| `df` | Degrees of freedom. |
| `p` | P-value. |
| `mean_diff` | Mean paired difference. |
| `se_diff` | Standard error of the mean difference. |
| `ci_lower` | Lower confidence interval bound. |
| `ci_upper` | Upper confidence interval bound. |
| `cohens_d` | Standardized paired mean difference. |
