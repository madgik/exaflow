## binned_mann_whitney.py - Binned Mann-Whitney U Test

### Name

**Binned Mann-Whitney U Test**

### Type

**Non-parametric statistics** (rank-based two-sample test)

### Goal (Why we need it)

Tests whether the distribution of a numerical variable differs between two
independent groups without assuming normality and without sharing raw values
across workers. Ranks are approximated via histogram binning — all values
within the same bin receive the same average rank — making the approach both
privacy-preserving and communication-efficient.

### When to use

Use `FederatedBinnedMannWhitneyUTest` when:

- you need a non-parametric two-sample comparison (distributions may not be normal)
- exact ranking across sites is not feasible (raw data must stay local)
- sample sizes are large enough for the normal approximation to hold (n ≥ 20 per group recommended)
- an approximate result is acceptable; use `num_bins ≥ 40` for reasonable accuracy

### When NOT to use

Avoid / be careful when:

- sample sizes are very small (n < 10 per group) — the asymptotic p-value degrades
- you need exact p-values — the binned approximation introduces rank error
- you need a multi-group comparison — this test handles exactly two groups

______________________________________________________________________

### Inputs / Outputs

| Item | Description |
| ------------------- | ---------------------------------------------------------------- |
| **x** | `np.ndarray` — group A observations (finite values only) |
| **y** | `np.ndarray` — group B observations (finite values only) |
| **alternative** | `'two-sided'`, `'less'`, or `'greater'` (default `'two-sided'`) |
| **use_continuity** | Apply continuity correction to z-score (default `True`) |
| **num_bins** | Number of histogram bins for rank approximation (default `40`) |
| **aggregator** | `NumpyAggregator` wrapping the federated agg client |

**Outputs** — dict with:

- `u_stat`: Mann-Whitney U statistic for group A.
- `p_value`: p-value for the selected alternative.
- `z_score`: continuity-corrected z-score of the U selected for the chosen
  alternative (group A's U for `greater`, the opposite U `n1 * n2 - u_stat` for
  `less`, the larger of the two for `two-sided`). It corresponds to `p_value`,
  not necessarily to the reported `u_stat`.
- `n1`: total observations in group A (across all workers).
- `n2`: total observations in group B (across all workers).

### How it works

`FederatedBinnedMannWhitneyUTest.compute(x, y, ...)`:

1. **Global bounds** — computes the global min and max of the concatenated
   sample (`x ∪ y`) via a single federated round-trip. This defines shared
   bin edges used at every worker.
1. **Federated histograms** — `SimpleHistogram.compute(x, num_bins, bounds=bounds)`
   and the same for `y`. Each call is one federated sum round-trip. Workers
   compute local histograms over the shared bin edges; the aggregation server
   sums them into global counts.
1. **Rank assignment** — for each bin `i` with total count `t_i`, the average
   rank is `previous_count + (t_i + 1) / 2`. All values in that bin receive
   this rank (the standard tied-rank convention).
1. **U statistic** — `U = n1 * n2 + n2 * (n2 + 1) / 2 - sum(ranks_y)`.
1. **Tie correction** — `sum(t^3 - t for t > 1) / (N * (N - 1))` reduces
   the variance estimate when many values share a bin.
1. **Normal approximation** — z-score with optional continuity correction,
   p-value from the standard normal survival/CDF.

Total federated round-trips: **3** (1 min/max + 2 histogram sums).

### Relationship with the Exareme3 wrapper

`BinnedMannWhitneyUTest` (exareme3) splits a DataFrame by a grouping variable
and delegates to this class. The federated primitive itself is agnostic to
DataFrame structure and accepts raw numpy arrays directly.
