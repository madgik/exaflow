# Descriptive Statistics

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

The mathematical method, library API, aggregation flow, numerical decisions, reference comparisons, and limitations are documented in [Descriptive summaries](../../exaflow/algorithms/federated/docs/describe.md).

## Application integration

The wrapper supplies `dataset` as the library grouping column and provides metadata category levels, integer-variable flags, and the configured minimum row count. It converts library group records into `featurewise` dataset records, labels the combined group `all datasets`, and supplies missing dataset/variable placeholders.

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
| `data.q1`, `data.q2`, `data.q3` | Quartiles for numerical variables. Per-dataset: exact (pandas). Combined: federated histogram-based estimate. |
| `data.median` | Alias for `q2` on the combined record; not set on per-dataset records. |
| `data.counts` | Category counts for nominal variables. |
