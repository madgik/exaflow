## Quartiles (Histogram-Based)

<b><h4>Aggregation Server</h4></b>
Some algorithms use the aggregation server to combine partial vectors (e.g., sums)
from workers into a single global result. The controller coordinates the flow and
workers send partial aggregates to the aggregation server via gRPC; the combined
result is then used in the algorithm's global step.

#### Overview

Estimates the three standard quartiles — **Q1 (25th percentile)**, **Q2 (median,
50th percentile)**, and **Q3 (75th percentile)** — of a numerical variable `y`
without sharing raw data across workers. The procedure builds a federated
histogram and iteratively refines the estimate by pushing only the selected bin's
data into the next round, re-binning over its own extent on every iteration.

Each result entry also returns `actual_q` — the cumulative fraction of data at or
below the returned value. It is `null` when the value falls exactly on the
requested quartile (no deviation to report), and a number otherwise — the
achieved fraction, which can exceed `q` when duplicate values make the discrete
quantile overshoot or when histogram binning approximates.

#### Parameters

- `y`: numerical variable (`REAL`/`INT`) to estimate the quartiles for. Accepts
  at most one variable (`max_count=1`).
- `num_bins`: integer in `[2, 100]`, default `20`. Number of bins used for each
  histogram refinement step.
- `max_iterations`: integer in `[1, 20]`, default `5`. Number of times the
  histogram is rebuilt and refined into the target bin.
- `threshold`: real in `[0, 0.1]`, default `0.001`. Early-stop threshold: if the
  cumulative fraction at a bin edge is within `threshold` of the target quartile,
  that edge is returned immediately without further refinement.

The quartile levels `[0.25, 0.5, 0.75]` are fixed and are not a user parameter.
The histogram strategy is also automatic: `INT` variables use Wilkinson
whole-number bin edges; `REAL` variables use equal-width bins.

#### Result

A list of three entries, one per quartile:

- `q`: the quartile level (0.25, 0.50, or 0.75).
- `value`: estimated value at that quartile.
- `actual_q`: `null` when `value` sits exactly at the requested quartile (its
  cumulative fraction equals `q`). Otherwise the true cumulative fraction of data
  at or below `value` — which exceeds `q` when duplicates make the discrete
  quantile overshoot (e.g. `y = [1, 1, 1, 2]` at `q=0.5` yields `value = 1`
  with `actual_q = 0.75`) or when histogram binning approximates.

Both `value` and `actual_q` are `null` when no finite data is available.

#### Approximation Example

Consider `y = [1, 2, 3, ..., 100]`.

- The exact quartiles are `25.75` (Q1), `50.5` (Q2), `75.25` (Q3).

- With `num_bins=20` and `max_iterations=5`, each quartile is refined
  independently through up to 5 histogram passes.

  **Pass 1** — full range `[1, 100]`, bin width ≈ 5:

  - For Q2 (`q=0.5`): target bin identified; only that bin's data is pushed
    into the next round.

  **Passes 2–5** — progressively narrower sub-ranges; after 5 passes the
  effective bin is ≈ `5 × (1/20)⁴ ≈ 3 × 10⁻⁴` wide.

  **Final estimates**: close to true quartiles; `actual_q` reports the exact
  cumulative fraction achieved (or `null` when it equals the target).

For distributions with a **large empty gap** (e.g. bimodal data), any value in
the gap is a valid quartile — different conventions will disagree, both being correct.

Constant (zero-range) data — including a sub-range that collapses to a repeated
value during refinement — short-circuits to the exact value rather than
interpolating.

#### Exareme3 Notes

- `y` accepts at most one variable (`max_count=1`).
- The three quartiles Q1/Q2/Q3 are always computed; there is no `q` parameter.
- Integer mode is derived automatically from the `sql_type` of `y`; no flag is
  needed from the caller.
- The algorithm performs `max_iterations` federated histogram round-trips per
  quartile. All three quartiles are computed within a single aggregation
  server session.
- `actual_q` is derived from the value's true empirical CDF — a federated count
  of data points at or below it. It is `null` for an exact landing and reports
  the achieved fraction otherwise, so `|actual_q - q|` (when `actual_q` is not
  `null`) is an exact measure of the deviation.

#### Algorithm Implementation

[Quartiles (Histogram-Based)](../../exaflow/algorithms/exareme3/statistics/quartiles.py)
