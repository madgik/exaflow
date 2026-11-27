# Outlier Report

Runtime wrapper: [outliers/outlier_report.py](../../exaflow/algorithms/exareme3/outliers/outlier_report.py).

The report reuses `OutlierWinsorizer.validate_outlier_configuration()` from
`outliers/outlier_winsorizer.py`; statistical calculations remain in the
federated `outliers` family.

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

The mathematical method, library API, aggregation flow, numerical decisions, reference comparisons, and limitations are documented in [Outlier bounds and reporting](../../exaflow/algorithms/federated/docs/outlier_report.md).

## Application integration

The wrapper validates numerical variable metadata and dictionary parameters, supplies the dataset grouping column and configured minimum row count, and maps group records into `featurewise` dataset records.

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
