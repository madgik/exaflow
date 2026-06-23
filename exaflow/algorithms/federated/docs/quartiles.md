## quartiles.py - Quartiles

### Name

**Quartiles (Histogram-Based)**

### Type

**Descriptive Statistics** (quartile estimation)

### Goal (Why we need it)

Estimates the **three quartiles** — Q1 (25th percentile), Q2 (median / 50th percentile),
and Q3 (75th percentile) — for a numerical variable in a federated setting by iteratively
refining a histogram-based estimate, without ever sharing raw values across workers.
Also returns `actual_q` per entry — the true cumulative fraction achieved — so the
caller can measure the approximation error.

### When to use

Use `QuartilesAlgorithm` (backed by the `Percentile` primitive) when:

- you need the standard quartile summary (Q1, Q2, Q3) of a numerical variable
- exact order statistics cannot be computed because raw data must stay local
- an approximate result with a reported error (`actual_q`) is acceptable

### When NOT to use

Avoid / be careful when:

- you need exact quartiles (this is a histogram-based approximation)
- the variable has very few finite observations (the estimate degrades)
- the distribution is bimodal with a large gap and the target quartile falls
  in the gap — any value in that gap is mathematically valid, but different
  conventions will disagree

______________________________________________________________________

### Inputs / Outputs

| Item | Description |
| ------------------- | ---------------------------------------------------------------- |
| **x** | `pandas.Series` of numerical values for this client |
| **num_bins** | Bin count used for each histogram refinement step (default 20) |
| **max_iterations** | Number of zoom/refine rounds (default 5) |
| **threshold** | Early-stop: stop if cumulative fraction is within this of the target quartile |
| **aggregator** | `NumpyAggregator` wrapping the federated agg client |

The quartile values `q = [0.25, 0.5, 0.75]` are fixed and not exposed as a
user parameter. The histogram strategy is determined automatically from the
`sql_type` of `y`: `INT` variables use Wilkinson whole-number bin edges;
`REAL` variables use equal-width bins.

**Outputs** (one entry per quartile)

- `q`: the quartile level (0.25, 0.50, or 0.75).
- `value`: estimated value at that quartile.
- `actual_q`: the true cumulative fraction of data at or below `value`. It is
  `null` when the value falls exactly on the requested quartile (no deviation),
  and a number otherwise — the achieved fraction, which exceeds `q` when
  repeated values make the discrete quantile overshoot or when histogram
  binning approximates.

### How it works

`QuartilesAlgorithm.run()` hardcodes `q_values = [0.25, 0.5, 0.75]` and calls
`Percentile.compute(x, q, ...)` once per quartile.

For **continuous** variables the call computes a federated `SimpleHistogram`
over the data range, walks the bins accumulating `total_counter`, and when
`total_counter / n` crosses the target `q` it pushes only that bin's data into
the next round — re-binned over its own extent (matching `np.histogram` bin
membership: half-open `[lo, hi)` for every bin except the last, which is closed)
— for up to `max_iterations` passes.

For **integer** variables the call uses `WilkinsonHistogram` with `min_step=1.0`
and repeatedly re-bins the bucket holding the target rank with whole-number
edges until it spans a single integer — the discrete quantile, the smallest
value with `CDF(value) >= q`.

In both cases `value` is paired with its true empirical CDF, obtained by a
federated count of the points at or below it. The public `Percentile.compute`
reports `actual_q` as `null` when that CDF equals the requested `q` (an exact
landing) and as the achieved fraction otherwise. Constant (zero-range) data —
including a sub-range that collapses to a repeated value during refinement — is
short-circuited to the exact value rather than interpolated.

All three quartiles are computed inside a single aggregation server session.
