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

Steps are organized by domain and identified by `PreprocessingStep` inheritance.
Recursive discovery exposes the complete set through the preprocessing registry
and `GET /specifications/preprocessing`; folder location does not affect their
availability.

| Step | Purpose | Runtime module |
| --- | --- | --- |
| Missing Values Handler | Drop or impute missing values per variable. | [missing_values/missing_values_handler.py](../../exaflow/algorithms/exareme3/missing_values/missing_values_handler.py) |
| Outlier Winsorizer | Clip numerical outliers using configured bounds. | [outliers/outlier_winsorizer.py](../../exaflow/algorithms/exareme3/outliers/outlier_winsorizer.py) |
| Categorical Column Creator | Create a categorical column from filter rules. | [feature_engineering/categorical_column_creator.py](../../exaflow/algorithms/exareme3/feature_engineering/categorical_column_creator.py) |
| Longitudinal Transformer | Convert two-visit records into first, second, or difference variables. | [longitudinal/longitudinal_transformer.py](../../exaflow/algorithms/exareme3/longitudinal/longitudinal_transformer.py) |
| KMeans Column Creator | Assign a categorical cluster using stored fitted centers. | [cluster/kmeans_cluster_creator.py](../../exaflow/algorithms/exareme3/cluster/kmeans_cluster_creator.py) |

Related reports and transformations share their domain: the outlier report
sits beside winsorization, and K-means sits beside its cluster column creator.
The report imports validation from the winsorizer that owns the clipping rules.
The cluster column creator imports replay models and fingerprint helpers from
K-means, which owns the fitted result contract.

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
| Column creators | `new_column_names` | Ordered names for the created columns. |
| KMeans Column Creator | `k`, `enumeration_prefix` | Fitted category count and prefix, populated from the reusable model. |
| KMeans Column Creator | `reusable_preprocessing` | Complete replay payload returned by a fitted K-means analysis. |

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

KMeans cluster creation replays the centers from a previous K-means analysis;
it does not fit a new model. It creates one categorical column with one category
per fitted cluster for downstream algorithms that accept categorical covariates.
The replay payload must match the data model, datasets, filters, and clustering
variables of the current analysis.

### Categorical column creation

Filter rules define categories for a new nominal text column. Each row must
match at most one rule; unmatched rows receive the configured default category
or null. Overlapping matches are rejected.

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
        assign rows to the nearest stored K-means center
        create a categorical cluster column

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
- KMeans cluster creation reuses stored centers and does not fit K-means again.
- KMeans cluster outputs are categorical covariates, not clinical diagnoses.
- Preprocessing order is defined by the request list. The server validates and
  executes steps in the submitted order.

## Outputs

Column creators declare an `outputs` object containing `new_column_names_param`
and ordered `columns`. Each definition produces one column unless its optional
`column_count_param` references a required, single-valued integer request
parameter. In that case it repeats by the supplied positive count. The names
parameter is a required list whose length equals the total expanded count.
Both existing creators produce one column and accept `new_column_names: ["chosen_name"]`.

Each column has a physical `type` and statistical `stattype`. Numerical columns
serialize `enumerations: null`. Fixed definitions serialize
`column_count_param: null`; unused enumeration-source references are also null.
Nominal columns require a source rule:
`dict_keys` uses `source_parameter_name` and can include an
`additional_value_parameter_name`; `prefixed_range` additionally requires
`prefix_parameter_name` and generates one-based categories from the count.
The categorical creator uses the keys of `rules` and any `default_enumeration`.
K-means uses its fitted count and prefix. See the
[API output contract](../api-specification.md#preprocessingoutputsspecificationdto)
for the complete schema and a mixed fixed/dynamic numerical example.

The controller and worker transformation hooks validate declared names, types,
and metadata consistently. Existing metadata fields are preserved. Fixed and
dynamic definitions can be mixed: consume one name per fixed entry and the
specified number per dynamic entry, in declaration order. All repetitions
inherit their definition's type and enumeration rule, using independent
metadata dictionaries. `min` is optional UI metadata for count parameters;
positive counts are enforced at runtime regardless. No literal `column_count`
field is supported. Existing steps that modify columns without creating
new ones do not declare this output container.

This replaces scalar `code` requests and the former `output` metadata. Consumers
must migrate together; no legacy aliases are provided.

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
