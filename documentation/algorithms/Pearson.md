# Pearson Correlation

## Overview

Pearson correlation measures linear association between numerical variables. The
algorithm computes correlations, p-values, and confidence intervals for all
requested variable pairs.

## Inputs

### Required inputs

| Input | Description |
|---|---|
| `y` | Numerical variables for the primary correlation axis. |
| `x` | Optional numerical variables for the secondary axis. |

When `x` is omitted, correlations are computed among the variables in `y`.

### Parameters

| Parameter | Description | Default |
|---|---|---|
| `alpha` | Confidence level used for correlation intervals. | `0.95` |

## Method

The mathematical method, library API, aggregation flow, numerical decisions, reference comparisons, and limitations are documented in [Pearson correlation](../../exaflow/algorithms/federated/docs/pearson_correlation.md).

## Outputs

| Field | Description |
|---|---|
| `title` | Result title. |
| `n_obs` | Number of observations used. |
| `correlations` | Matrix of Pearson correlations. |
| `p_values` | Matrix of p-values. |
| `ci_lo` | Matrix of lower confidence interval bounds. |
| `ci_hi` | Matrix of upper confidence interval bounds. |
