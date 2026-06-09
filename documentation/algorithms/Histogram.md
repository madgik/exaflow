# Histogram

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

For numerical variables, the algorithm supports two binning strategies:

- `simple`: equal-width bins between the aggregated minimum and maximum.
- `wilkinson`: bin boundaries are snapped to readable 1/2/5 x 10^n step sizes,
  and the range is extended outward to those boundaries.

For integer-valued numerical variables with `wilkinson` binning, the step size
is floored at `1.0`, so adjacent integer values are not split by sub-integer bin
edges.

For categorical variables, bins are category labels. Metadata enumerations are
reported first in their declared order. Any observed categories not present in
metadata are appended. When no enumeration is available, globally observed
categories are discovered and sorted.

When grouping variables are provided, each grouping level receives a histogram
using the same numerical bin edges or categorical bin order as the base
histogram.

## Federated computation

The histogram is computed without sharing row-level data. Each site contributes
local extrema, category levels, and bin-count vectors. Aggregated quantities are
used to build aligned bins and total counts.

### Aggregated quantities

| Quantity | Purpose |
|---|---|
| Numerical minimum and maximum | Define shared numerical bin ranges. |
| Category levels | Align categorical bins across sites. |
| Group levels | Align grouped histogram rows. |
| Bin counts | Compute aggregate histogram counts. |
| Group-specific count matrices | Compute counts for each group level and bin. |

### Federated flow

```text
Input:
    y: variable to summarize
    x: optional grouping variables
    bins: requested numerical bin count
    histogram_type: numerical binning strategy

Step 1:
    Remove rows with missing values in y or the selected grouping variables.

Step 2:
    If y is categorical:
        discover observed category levels
        merge them with metadata enumerations when available
        each site counts observations per category

Step 3:
    If y is numerical:
        aggregate the global minimum and maximum
        build simple or Wilkinson bin edges
        each site counts observations in the shared bins

Step 4:
    For each grouping variable:
        align group levels from metadata or observed values
        compute local group-by-bin count matrices

Step 5:
    Aggregate base counts and group-specific count matrices.

Step 6:
    Mask counts below the minimum row-count threshold.

Output:
    base histogram and optional grouped histograms
```

## Technical decisions

- The required missing-values preprocessing step removes missing values before
  histogram counts are computed.
- `y` accepts at most one variable.
- Numerical `bins` is rounded to an integer and bounded by the public
  specification range.
- If a simple numerical histogram has identical minimum and maximum, the maximum
  is increased by `1.0` to create a valid bin range.
- Wilkinson binning may return a different number of bins than requested because
  the requested value is treated as a hint for selecting readable boundaries.
- Category and group metadata enumerations preserve their declared order.
- Counts lower than the minimum row-count threshold are returned as `null`.

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

## Validation against state-of-the-art implementation

Numerical counting is aligned with `numpy.histogram` after shared bin edges are
defined. Standalone tests cover simple numerical histograms, Wilkinson binning,
integer-aware Wilkinson boundaries, categorical histograms, and grouped
histograms.

## Limitations and assumptions

- The output reports counts, not densities or percentages.
- Numerical bin edges are derived from aggregated extrema, so extreme values can
  affect bin width.
- Counts below the privacy threshold are masked and cannot be distinguished from
  absent low-count bins in the response.
- Grouped histograms can be sparse when many grouping levels are selected.
- Wilkinson binning prioritizes readable boundaries over preserving the exact
  requested number of bins.
