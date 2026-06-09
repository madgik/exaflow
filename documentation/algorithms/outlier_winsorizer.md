# Outlier Winsorizer

## Table of contents

- [Overview](#overview)
- [Inputs](#inputs)
  - [Required inputs](#required-inputs)
  - [Parameters](#parameters)
- [Preprocessing method](#preprocessing-method)
- [Federated computation](#federated-computation)
  - [Aggregated quantities](#aggregated-quantities)
  - [Federated flow](#federated-flow)
- [Technical decisions](#technical-decisions)
- [Outputs](#outputs)
- [Validation against state-of-the-art implementation](#validation-against-state-of-the-art-implementation)
- [Limitations and assumptions](#limitations-and-assumptions)

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

## Preprocessing method

For each configured numerical variable, the step computes lower and/or upper
bounds and clips values outside those bounds:

```text
clipped_value = min(max(value, lower_bound), upper_bound)
```

Bounds are defined by the same strategies documented in
[Outlier Report](outlier_report.md).

## Federated computation

This preprocessing step does not compute combined global bounds. Bounds are
computed independently where data are held, and clipping is applied before the
downstream algorithm receives the data.

### Aggregated quantities

| Quantity | Purpose |
|---|---|
| Per-site numerical values | Compute local clipping bounds. |
| Minimum row-count threshold | Validate enough non-missing values before clipping. |

No cross-site numerical moments or quantiles are aggregated by this step.

### Federated flow

```text
Input:
    selected data
    strategies, tails, folds

Step 1:
    Validate that configured variables exist and are numerical.

Step 2:
    Convert configured variables to numeric values.

Step 3:
    Drop rows with missing values in configured variables.

Step 4:
    For each configured variable:
        validate minimum non-missing row count
        compute local bounds using the selected strategy
        clip values to the selected lower and/or upper bound

Step 5:
    Update metadata min/max for clipped variables when bounds exist.

Output:
    transformed data passed to the downstream algorithm
```

## Technical decisions

- Bounds are local to the data partition, not combined global bounds.
- Configured integer variables are promoted to real metadata after clipping.
- The step drops rows with missing values in configured variables before
  clipping.
- `left` clips only the lower tail, `right` clips only the upper tail, and
  `both` clips both tails.
- Invalid folds are rejected before data transformation.

## Outputs

| Field | Description |
|---|---|
| Transformed data | Data with configured numerical variables clipped. |
| Transformed metadata | Metadata with updated `min` and/or `max` for clipped variables. |

## Validation against state-of-the-art implementation

The clipping rules are aligned with common winsorization practice using
Gaussian, IQR, MAD, and quantile bounds. Tests validate parameter checks,
missing-value behavior, metadata updates, and expected downstream algorithm
behavior.

## Limitations and assumptions

- Only numerical variables are supported.
- Bounds can differ by site because they are computed independently.
- Quantile clipping uses local quantiles.
- Clipping can change downstream model estimates and descriptive summaries.
