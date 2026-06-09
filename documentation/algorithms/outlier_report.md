# Outlier Report

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

Outlier Report inspects numerical variables and reports outlier bounds, counts,
and percentages per dataset. It supports Gaussian, IQR, MAD, and quantile-based
screening rules.

## Inputs

### Required inputs

| Input | Description |
|---|---|
| `y` | Numerical variables to inspect. |
| `x` | Optional additional numerical variables to inspect. |

### Parameters

| Parameter | Description | Default |
|---|---|---|
| `strategies` | Dictionary mapping each variable to `gaussian`, `iqr`, `mad`, or `quantile`. | Required |
| `tails` | Optional dictionary mapping each variable to `left`, `right`, or `both`. | `both` |
| `folds` | Optional dictionary of strategy-specific threshold folds. | Strategy default |

Default folds:

| Strategy | Default fold | Rule |
|---|---:|---|
| `gaussian` | `3.0` | `mean +/- fold * sample_std` |
| `iqr` | `1.5` | `Q1 - fold * IQR`, `Q3 + fold * IQR` |
| `mad` | `3.0` | `median +/- fold * 1.4826 * MAD` |
| `quantile` | `0.05` | `fold` and `1 - fold` quantiles |

## Method

For each variable and dataset, the selected rule defines lower and/or upper
bounds. Values below the lower bound or above the upper bound are counted as
outliers, depending on the selected tail.

## Federated computation

The report does not aggregate bounds across datasets. Bounds and outlier counts
are computed per dataset where the data reside, and only summary records are
returned.

### Aggregated quantities

| Quantity | Purpose |
|---|---|
| Per-dataset numerical values | Compute local bounds and counts. |
| Minimum row-count threshold | Suppress reports with too few non-missing values. |

No cross-site numerical moments or quantiles are aggregated for this report.

### Federated flow

```text
Input:
    variables: numerical variables
    strategies: outlier rule per variable
    tails: optional inspected tail per variable
    folds: optional threshold per variable

Step 1:
    Validate that configured variables are numerical.

Step 2:
    Resolve default folds for variables without explicit fold values.

Step 3:
    For each dataset and variable:
        drop missing and non-numeric values
        validate minimum non-missing row count
        compute bounds using the selected strategy
        count lower and upper outliers
        mask small non-zero counts

Output:
    per-variable, per-dataset outlier report records
```

## Technical decisions

- Bounds are dataset-level, not combined across all datasets.
- Quantile bounds are dataset-level quantiles.
- `gaussian`, `iqr`, and `mad` folds must be positive finite numbers.
- `quantile` folds must be in `(0, 0.5)`.
- A zero outlier count is reported as `0`; a non-zero count below the minimum
  row-count threshold is reported as `null`.
- If either side is suppressed, total outlier count and percentage are also
  suppressed.

## Outputs

| Field | Description |
|---|---|
| `featurewise` | List of outlier report records. |
| `variable` | Variable inspected. |
| `dataset` | Dataset label. |
| `data.strategy` | Outlier rule used. |
| `data.tail` | Tail inspected. |
| `data.fold` | Fold threshold used. |
| `data.lower_bound`, `data.upper_bound` | Computed bounds, when applicable. |
| `data.lower_outlier_count`, `data.upper_outlier_count` | Outlier counts by side, possibly masked. |
| `data.total_outlier_count` | Total outlier count, possibly masked. |
| `data.total_outlier_percentage` | Percentage of non-missing observations flagged as outliers. |

## Validation against state-of-the-art implementation

The rules are aligned with standard Gaussian, IQR, MAD, and quantile screening
methods. Tests validate parameter handling, privacy masking, and expected report
fixtures rather than comparing to a single external package.

## Limitations and assumptions

- Only numerical variables are supported.
- Bounds are computed per dataset and can differ across datasets.
- Quantile, IQR, and MAD rules are descriptive screening methods, not formal
  hypothesis tests.
- Small non-zero outlier counts can be suppressed.
