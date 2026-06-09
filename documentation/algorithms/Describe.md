# Descriptive Statistics

## Table of contents

- [Overview](#overview)
- [Inputs](#inputs)
  - [Required inputs](#required-inputs)
  - [Parameters](#parameters)
- [Method](#method)
- [Federated computation](#federated-computation)
  - [Aggregated quantities](#aggregated-quantities)
  - [Federated flow](#federated-flow)
- [Technical decisions](#technical-decisions)
- [Outputs](#outputs)
- [Validation against state-of-the-art implementation](#validation-against-state-of-the-art-implementation)
- [Limitations and assumptions](#limitations-and-assumptions)

## Overview

Descriptive statistics summarize selected numerical and nominal variables. The
result contains per-dataset summaries and a combined summary across all selected
datasets.

## Inputs

### Required inputs

| Input | Description |
|---|---|
| `y` | Variables to summarize. |
| `x` | Optional additional variables to summarize. |

### Parameters

No user parameters are exposed for this algorithm.

## Method

For numerical variables, the algorithm reports counts, missing-value counts,
mean, sample standard deviation, minimum, quartiles, median, and maximum where
available.

For nominal variables, it reports counts, missing-value counts, total rows, and
category counts.

## Federated computation

The summaries are computed without sharing row-level data. Dataset-level
summaries are computed at each site subject to a minimum row-count threshold.
Combined summaries are computed from aggregated sufficient statistics and
category counts.

### Aggregated quantities

| Quantity | Purpose |
|---|---|
| Non-missing counts | Report valid observations and compute means. |
| Missing counts | Report missingness. |
| Total row counts | Report denominator per variable and dataset. |
| Numerical sums and sums of squares | Compute combined means and sample standard deviations. |
| Numerical minima and maxima | Compute combined ranges. |
| Nominal category counts | Compute combined category frequencies. |

### Federated flow

```text
Input:
    variables: numerical and/or nominal variables
    datasets: selected datasets

Step 1:
    For each site and dataset:
        split variables into numerical and nominal groups
        check minimum row-count threshold

Step 2:
    For numerical variables:
        compute local count, missing count, total count
        compute mean, sample standard deviation, min, quartiles, median, max
        store sum and sum of squares for aggregation

Step 3:
    For nominal variables:
        compute local count, missing count, total count
        compute category counts

Step 4:
    Aggregate sufficient statistics and category counts for the combined summary.

Step 5:
    Append placeholder records for selected dataset/variable combinations that
    have no reportable data.

Output:
    feature-wise dataset summaries and combined summaries
```

## Technical decisions

- Dataset-level records below the minimum row-count threshold return `null`
  summary data.
- Numerical quartiles are reported for dataset-level summaries; combined
  summaries do not aggregate quantiles.
- Combined numerical standard deviations use aggregated sums and sums of
  squares with sample-variance denominator `n - 1`.
- Nominal combined summaries use metadata-defined category levels and omit
  zero-count levels.
- Missing values are handled independently for each variable.

## Outputs

| Field | Description |
|---|---|
| `featurewise` | List of variable-by-dataset summary records. |
| `variable` | Variable name. |
| `dataset` | Dataset label, including `all datasets` for combined summaries. |
| `data.num_dtps` | Number of non-missing values. |
| `data.num_na` | Number of missing values. |
| `data.num_total` | Total number of rows considered. |
| `data.mean`, `data.std`, `data.min`, `data.max` | Numerical summaries. |
| `data.q1`, `data.q2`, `data.q3` | Dataset-level quartiles for numerical variables. |
| `data.counts` | Category counts for nominal variables. |

## Validation against state-of-the-art implementation

Standalone tests compare numerical summary behavior with
`statsmodels.stats.weightstats.DescrStatsW` and verify category/count aggregation
against expected fixtures.

Reference behavior is aligned with standard descriptive-statistics formulas,
with combined summaries computed from aggregated sufficient statistics.

## Limitations and assumptions

- This is descriptive only; no hypothesis tests are performed.
- Combined quantiles are not computed from all row-level values.
- Suppressed dataset-level summaries are represented with `null` data.
- Category counts depend on available metadata levels for the combined summary.
