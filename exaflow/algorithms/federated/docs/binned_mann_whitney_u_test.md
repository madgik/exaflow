# Binned Mann–Whitney U test

Implementation: [statistics/binned_mann_whitney_u_test.py](../statistics/binned_mann_whitney_u_test.py).

The Binned Mann-Whitney U test is a non-parametric test that assesses whether
the distribution of a numerical variable differs between two independent groups.
Unlike the independent t-test, it makes no assumption about normality and
operates on ranks rather than raw values.

This implementation is a **binned approximation**: rather than computing exact
ranks across the full dataset, it groups values into histogram bins and assigns
the average rank of each bin to all values inside it. This makes exact rank
computation unnecessary — no site ever shares individual values — while
retaining the test's non-parametric properties.

The U statistic and p-value are derived from the normal approximation, matching
the approach of `scipy.stats.mannwhitneyu` with `method='asymptotic'`.

## Library interface

`FederatedBinnedMannWhitneyUTest(aggregator).compute(x, y, alternative="two-sided", use_continuity=True, num_bins=40)` compares two numerical arrays. The result contains `u_stat`, `z_score`, `p_value`, `n1`, and `n2`. Histogram ranks make this an approximation. Nonfinite values are removed by the test. Callers split the two groups before invoking it.

## Statistical method

The null hypothesis is:

```text
H0: the distributions of y in group A and group B are identical
```

The U statistic is computed as:

```text
U = n1 * n2 + n2 * (n2 + 1) / 2 - sum(ranks_B)
```

Where `ranks_B` are the approximate ranks assigned to group B observations via
histogram binning. The z-score with continuity correction is:

```text
mu    = n1 * n2 / 2
sigma = sqrt(n1 * n2 * (N + 1 - tie_correction) / 12)
z     = (U - mu ± 0.5) / sigma
```

The tie correction accounts for values sharing the same histogram bin:

```text
tie_correction = sum(t^3 - t for each bin with count t > 1) / (N * (N - 1))
```

The p-value is derived from the standard normal distribution.

## Federated computation

The test is computed without sharing row-level data. Each site contributes
only aggregate counts and extrema; individual rows are not transferred.

### Aggregated quantities

| Quantity | Purpose |
|---|---|
| Global min and max of combined groups | Define shared bin boundaries used at all sites. |
| Histogram bin counts for group A | Approximate ranks for group A observations. |
| Histogram bin counts for group B | Approximate ranks for group B observations. |
| Global counts n1 and n2 | Compute U statistic, mu, and sigma. |

### Federated flow

```text
Receive the two numerical samples x and y and remove nonfinite observations.
Aggregate group counts and determine shared extrema/bin edges.
Compute each group's local histogram and sum bin counts.
Assign average ranks to bins and compute U for the first sample.
Apply bin-level tie correction and the selected continuity/tail convention.
Return u_stat, p_value, z_score, n1, and n2.
```

## Numerical decisions

- Ranks are approximated at the bin level: all values in the same bin receive
  the same average rank. Accuracy increases with `num_bins`. The default of 40
  provides a good balance between accuracy and communication cost.
- The normal approximation (asymptotic method) is used. Exact permutation-based
  p-values are not implemented by this method.
- Continuity correction is enabled by default and can be disabled with `use_continuity=False`, consistent with
  `scipy.stats.mannwhitneyu(use_continuity=True)`.
- Bin boundaries are shared across all sites so that histogram counts are
  directly additive (no alignment needed at aggregation time).
- The `INT` vs `REAL` distinction does not affect the algorithm — both types
  use equal-width bins over the observed range.
- Nonfinite observations are removed from each supplied sample before counts and ranks are computed.
- `u_stat` is the U statistic for the first sample; `z_score` follows the U tail selected for `alternative`, so the two need not describe the same U for a one-sided test.

## Reference comparison

Standalone tests compare the result with `scipy.stats.mannwhitneyu`:

```python
scipy.stats.mannwhitneyu(
    group_a,
    group_b,
    alternative=alternative,
    method="asymptotic",
    use_continuity=True,
)
```

Tests use `num_bins=100` and sample sizes of n ≥ 100 per group to minimise
binning approximation error. The tolerance is `atol=0.05` on p-values.
All tests also verify that both methods reach the same significance conclusion
at α = 0.05.

Test scenarios cover: clearly different normal distributions, identical
distributions, directional alternatives (`less`, `greater`), unequal sample
sizes, skewed (exponential) distributions, integer-valued outcomes, negative
ranges, and large variance differences.

## Limitations

- The p-value is approximate due to histogram rank binning. Accuracy degrades
  with very few bins or very small sample sizes.
- The normal approximation assumes sufficiently large samples. For very small
  groups (n < 10) the asymptotic p-value may not be reliable.
- Exactly two groups are compared. Multi-group comparisons are not supported.
- The two groups are assumed independent.
