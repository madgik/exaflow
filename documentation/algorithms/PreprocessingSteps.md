# Preprocessing Steps

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

Preprocessing steps transform selected variables before an algorithm is fitted or
evaluated. They are configured per request and run before the main statistical or
machine-learning method.

Implemented preprocessing steps include:

| Step | Purpose |
|---|---|
| Missing Values Handler | Drop or impute missing values per variable. |
| Outlier Winsorizer | Clip numerical outliers using configured bounds. |
| Longitudinal Transformer | Convert two-visit longitudinal records into first, second, or difference variables. |
| KMeans Cluster Creator | Create a categorical cluster covariate from numerical baseline variables. |

## Inputs

### Required inputs

Preprocessing steps operate on the variables selected by the downstream
algorithm. The longitudinal transformer also requires dataset, subject, and
visit identifiers so records can be matched across visits.

### Parameters

| Step | Parameter | Description |
|---|---|---|
| Missing Values Handler | `strategies` | Per-variable strategy: `drop`, `mean`, `median`, `most_frequent`, or `constant`. |
| Missing Values Handler | `fill_values` | Replacement values for variables using `constant`. |
| Outlier Winsorizer | `strategies` | Per-variable outlier clipping strategy. |
| Outlier Winsorizer | `tails` | Per-variable tail selection. |
| Outlier Winsorizer | `folds` | Per-variable clipping threshold. |
| Longitudinal Transformer | `visit1`, `visit2` | Visit identifiers to align. |
| Longitudinal Transformer | `strategies` | Per-variable strategy: `first`, `second`, or `diff`. |
| KMeans Cluster Creator | `cluster_variables` | Numerical variables used to fit K-means. |
| KMeans Cluster Creator | `k_selection` | `manual` or `elbow`. |
| KMeans Cluster Creator | `k`, `k_min`, `k_max` | Cluster-count settings. |
| KMeans Cluster Creator | `init_method`, `n_init` | Random-range initialization strategy and restart count. |
| KMeans Cluster Creator | `output_mode` | `full`, `binary`, or `subset`. |
| KMeans Cluster Creator | `binary_cluster` | Cluster used for binary `yes` / `no` output. |
| KMeans Cluster Creator | `selected_clusters` | Clusters kept explicitly in subset mode. |

## Preprocessing method

### Missing values

Missing values can be dropped or replaced. Mean, median, and most-frequent
imputation are computed from the data available at each site. Constant
imputation uses user-provided scalar values.

### Outlier winsorization

Numerical variables can be clipped with Gaussian, IQR, MAD, or quantile bounds.
See [Outlier Winsorizer](outlier_winsorizer.md) for the detailed clipping
rules.

### Longitudinal transformation

For two selected visits, records are matched by dataset and subject identifier.
Variables can retain the first visit value, retain the second visit value, or
use the difference:

```text
diff = value_at_visit2 - value_at_visit1
```

For the `diff` strategy, variable values are transformed to `visit2 - visit1`,
but the original variable code is preserved.

### KMeans cluster creation

KMeans cluster creation fits federated K-means on selected numerical variables
and creates a new categorical column. It is intended for downstream algorithms
that accept categorical covariates, such as linear regression, logistic
regression, and Cox regression.

The initialization method can be `random_range` or
`multi_start_random_range`. Multi-start fits several random-range
initializations and keeps the one with the lowest global inertia before creating
the derived categorical column.

Output modes:

| Mode | Categories | Meaning |
|---|---|---|
| `full` | `cluster_0`, `cluster_1`, ..., `cluster_k-1` | Every fitted cluster is exposed as a category. |
| `binary` | `yes`, `no` | `yes` means the row belongs to the selected cluster; `no` means any other cluster. |
| `subset` | selected clusters plus `other` | Selected clusters are kept explicit; all other clusters are combined. |

When `subset` has exactly one selected cluster, the output is automatically
binary. This is useful when the analysis question is membership in one
clinically reviewed cluster.

## Federated computation

Preprocessing is applied without sharing row-level data. Most transformations
are computed independently at each site. Steps that require alignment, such as
longitudinal matching, use the requested fixed identifier columns.

### Aggregated quantities

| Quantity | Purpose |
|---|---|
| Local means, medians, and modes | Impute missing values with per-site summaries. |
| Local outlier bounds | Clip numerical variables before downstream computation. |
| Dataset, subject, and visit identifiers | Match records for longitudinal transformations. |
| Updated metadata | Preserve transformed variable types, names, and ranges. |

### Federated flow

```text
Input:
    selected variables
    preprocessing configuration

Step 1:
    Validate preprocessing parameters against selected variables and metadata.

Step 2:
    Apply missing-value strategies:
        drop rows or fill missing values
        validate constant fill values

Step 3:
    Apply outlier winsorization when configured:
        compute local bounds
        clip selected numerical variables
        update metadata bounds

Step 4:
    Apply longitudinal transformation when configured:
        filter to the selected visits
        match records by subject and dataset
        compute first, second, or difference variables
        update variable names and metadata

Step 5:
    Apply KMeans cluster creation when configured:
        fit federated K-means on selected numerical variables
        create a categorical cluster column
        validate privacy for exposed categories

Step 6:
    Pass transformed data and metadata to the downstream algorithm.

Output:
    transformed data, transformed variable selections, and transformed metadata
```

## Technical decisions

- Missing-value imputation statistics are local, not combined global statistics.
- Mean and median imputation are restricted to numerical variables.
- Constant categorical fills must be compatible with metadata categories.
- Outlier winsorizer bounds are local and can differ across sites.
- Longitudinal `diff` is restricted to numerical variables.
- Longitudinal transformation requires exactly one strategy for every selected
  `x` and `y` variable.
- KMeans cluster creation requires an aggregation server because cluster centers
  and labels depend on global federated fitting.
- KMeans cluster outputs are categorical covariates, not clinical diagnoses.
- KMeans cluster creation rejects outputs where any exposed category is below
  the privacy minimum-row threshold.
- Preprocessing order is defined by the request list. The server validates and
  executes steps in the submitted order.

## Outputs

| Field | Description |
|---|---|
| Transformed data | Rows and values after preprocessing. |
| Transformed `x` and `y` | Variable selections after renaming or filtering. |
| Transformed metadata | Metadata updated for type promotions, bounds, and renamed variables. |

## Validation against state-of-the-art implementation

Missing-value behavior is aligned with common imputation strategies such as
`sklearn.impute.SimpleImputer`, with the important distinction that summary
statistics are computed per site. Winsorization is aligned with standard
Gaussian, IQR, MAD, and quantile clipping rules. Longitudinal transformation is a
deterministic record-alignment and arithmetic transformation validated by
focused preprocessing and downstream algorithm tests.

## Limitations and assumptions

- Preprocessing can change the estimand of downstream statistical tests.
- Local imputation and local winsorization can produce different transformed
  values at different sites.
- Longitudinal transformation requires reliable subject and visit identifiers.
- Difference transformations require numerical variables.
- Preprocessing should be chosen to match the scientific interpretation of the
  downstream analysis.
