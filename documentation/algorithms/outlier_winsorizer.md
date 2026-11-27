# Outlier Winsorizer

Runtime wrapper: [outliers/outlier_winsorizer.py](../../exaflow/algorithms/exareme3/outliers/outlier_winsorizer.py).

This step owns `validate_outlier_configuration()`, which the adjacent outlier
report also uses to validate the same winsorization rules.

## Overview

Outlier Winsorizer is a preprocessing step that clips selected numerical
variables to bounds computed from Gaussian, IQR, MAD, or quantile rules.

## Inputs

### Required inputs

The step acts on variables selected by the downstream algorithm and configured in
the `strategies` dictionary.

### Parameters

| Parameter | Description | Default |
|---|---|---|
| `strategies` | Dictionary mapping each variable to `gaussian`, `iqr`, `mad`, or `quantile`. | Required |
| `tails` | Optional dictionary mapping each variable to `left`, `right`, or `both`. | `both` |
| `folds` | Optional dictionary of strategy-specific clipping thresholds. | Strategy default |

## Method

See the [library method](../../exaflow/algorithms/federated/docs/outlier_report.md) and the bounds helper for the statistical calculation and core contracts.

## Application integration

The step validates numerical metadata and configured rules, removes rows missing the configured variables, computes local bounds through the shared helper, and clips the dataframe before downstream execution. `left`, `right`, and `both` select the tails to clip. Integer variables are promoted to real metadata. The library helper computes bounds and counts; clipping and metadata changes are implemented by this step.

## Outputs

| Field | Description |
|---|---|
| Transformed data | Data with configured numerical variables clipped. |
| Transformed metadata | Metadata with updated `min` and/or `max` for clipped variables. |
