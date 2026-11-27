# Histogram

## Overview

Histogram computes counts for one numerical or categorical variable, optionally
split by categorical grouping variables. Counts below the configured minimum
row-count threshold are masked as `null`.

## Inputs

### Required inputs

| Input | Description |
|---|---|
| `y` | One numerical or categorical variable to summarize. |
| `x` | Optional categorical grouping variables for grouped histograms. |

### Parameters

| Parameter | Description | Default |
|---|---|---|
| `bins` | Bin count used for numerical histograms. Ignored for categorical variables. | `20` |
| `histogram_type` | Binning strategy for numerical histograms: `simple` or `wilkinson`. Ignored for categorical variables. | `simple` |

## Method

The mathematical method, library API, aggregation flow, numerical decisions, reference comparisons, and limitations are documented in [Histograms](../../exaflow/algorithms/federated/docs/histogram.md).

## Application integration

Required missing-value preprocessing runs before counting. The wrapper derives integer mode from variable metadata and supplies the configured minimum row count. The request defaults to `simple`; the grouped library facade defaults to `wilkinson`, so the wrapper passes the requested strategy explicitly.

## Outputs

The response contains a `histogram` list. The first item is the base histogram
for `y`. Additional items represent grouped histograms, one item per grouping
variable level.

| Field | Description |
|---|---|
| `histogram` | List of base and grouped histogram result items. |
| `var` | Variable summarized. |
| `grouping_var` | Grouping variable name, or `null` for the base histogram. |
| `grouping_enum` | Group level, or `null` for the base histogram. |
| `bins` | Numerical bin edges or categorical labels. |
| `counts` | Counts per bin, with masked counts returned as `null`. |

For numerical histograms, `bins` contains bin edges and therefore has one more
entry than `counts`. For categorical histograms, `bins` contains category labels
and has the same length as `counts`.
