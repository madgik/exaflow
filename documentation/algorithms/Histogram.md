## Histogram

<b><h4>Aggregation Server</h4></b>
Some algorithms use the aggregation server to combine partial vectors (e.g., sums)
from workers into a single global result. The controller coordinates the flow and
workers send partial aggregates to the aggregation server via gRPC; the combined
result is then used in the algorithm's global step.

#### Overview

Computes a federated histogram for a target variable `y`, optionally grouped by
one or more categorical variables `x`. Workers compute local bin counts and the
aggregation server returns the global counts in a single round per histogram.
Counts below the privacy threshold are masked as `null`.

The algorithm supports three binning strategies:

- **Simple** (default) — equal-width bins for numerical targets between the
  global min and max. Selected via `histogram_type = "simple"`.
- **Wilkinson** — produces "nice-number" bin boundaries for numerical targets,
  chosen so that bin edges are easy to read. Selected via
  `histogram_type = "wilkinson"`. For integer-valued targets (detected from
  metadata `sql_type = int`), the bin step is floored at `1` and bin edges
  are whole numbers, so a single bin never splits adjacent integers apart.
  For example, an integer column ranging `0`–`7` with `bins = 20` requested
  would naturally snap to a `0.2` step (35 bins, cutting individual
  integers into pieces); the floor instead produces a step of `1` with
  edges `0, 1, …, 8` (one bin per integer value).
- **Categorical** — used automatically when `y` is a nominal variable. When the
  variable has enumerations declared in metadata, the order from metadata is
  honoured; otherwise the global category set is discovered via a federated
  union.

When grouping variables `x` are provided, one histogram is returned per group
level (with the same bin edges as the ungrouped histogram for numerical `y`).

#### Parameters

- `y`: target variable to bin. Numerical (`REAL`/`INT`) or categorical (`TEXT`).
- `x`: optional list of categorical grouping variables (`INT`/`TEXT`,
  `NOMINAL`).
- `bins`: integer in `[1, 100]`, default `20`. Number of bins for numerical
  targets; ignored for categorical targets.
- `histogram_type`: `"simple"` or `"wilkinson"`, default `"simple"`. Ignored
  for categorical targets.

#### Result

- `bins`: list of bin edges (numerical) or category labels (categorical).
- `counts`: per-bin global counts, with values below the worker's
  `minimum_row_count` privacy threshold replaced by `null`.
- `grouped`: per-grouping-variable, per-group counts, using the same bins as
  the ungrouped histogram.

#### Exareme3 Notes

- `y` accepts at most one variable (`max_count=1`); `x` accepts any number of
  optional grouping variables.
- Rows with `NA` in the selected columns (`y` and any chosen `x`) are dropped
  before computation.
- Enforces a minimum row count per worker via the worker privacy config
  (`worker.privacy.minimum_row_count`).
- For categorical targets without metadata enumerations, the global category
  set is discovered with a single `fed_union` call before counting.
- Grouped histograms reuse the bin edges established by the ungrouped
  histogram, ensuring consistency across groups.

#### Algorithm Implementation

[Histogram](../../exaflow/algorithms/exareme3/statistics/histogram.py)
