# Binned Mann-Whitney U Test

## Table of contents

- [Overview](#overview)
- [Inputs](#inputs)
  - [Required inputs](#required-inputs)
  - [Parameters](#parameters)
- [Statistical model](#statistical-model)
- [Federated computation](#federated-computation)
  - [Aggregated quantities](#aggregated-quantities)
  - [Federated flow](#federated-flow)
- [Technical decisions](#technical-decisions)
- [Outputs](#outputs)
- [Validation against state-of-the-art implementation](#validation-against-state-of-the-art-implementation)
- [Limitations and assumptions](#limitations-and-assumptions)

## Overview

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

## Inputs

### Required inputs

| Input | Description |
|---|---|
| `y` | Numerical outcome variable (`REAL` or `INT`). |
| `x` | Categorical grouping variable. |

### Parameters

| Parameter | Description | Default |
|---|---|---|
| `groupA` | First group category. | Required |
| `groupB` | Second group category. | Required |
| `alt_hypothesis` | Alternative hypothesis: `two-sided`, `less`, or `greater`. | `two-sided` |
| `num_bins` | Number of histogram bins used to approximate ranks. Higher values improve accuracy. | `40` |

## Statistical model

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
only histogram bin counts, which cannot be used to reconstruct individual values.

### Aggregated quantities

| Quantity | Purpose |
|---|---|
| Global min and max of combined groups | Define shared bin boundaries used at all sites. |
| Histogram bin counts for group A | Approximate ranks for group A observations. |
| Histogram bin counts for group B | Approximate ranks for group B observations. |
| Global counts n1 and n2 | Compute U statistic, mu, and sigma. |

### Federated flow

```text
Input:
    y: numerical outcome
    x: grouping variable
    groupA, groupB: selected categories
    num_bins: histogram resolution
    alternative: two-sided, less, or greater

Step 1:
    At each site:
        extract group A and group B observations
        compute local min and max of the combined sample

Step 2:
    Aggregate to find global min and max — defines shared bin edges.

Step 3:
    At each site:
        compute group A histogram over shared bin edges
        compute group B histogram over shared bin edges

Step 4:
    Aggregate bin counts for group A and group B.

Step 5:
    From global bin counts:
        assign average bin rank to all observations in each bin
        compute U statistic, tie correction, sigma, z-score, and p-value

Output:
    u_stat, p_value, z_score, n1, n2
```

## Technical decisions

- Ranks are approximated at the bin level: all values in the same bin receive
  the same average rank. Accuracy increases with `num_bins`. The default of 40
  provides a good balance between accuracy and communication cost.
- The normal approximation (asymptotic method) is used. Exact permutation-based
  p-values are not feasible in the federated setting.
- Continuity correction is always applied, consistent with
  `scipy.stats.mannwhitneyu(use_continuity=True)`.
- Bin boundaries are shared across all sites so that histogram counts are
  directly additive (no alignment needed at aggregation time).
- The `INT` vs `REAL` distinction does not affect the algorithm — both types
  use equal-width bins over the observed range.

## Outputs

| Field | Description |
|---|---|
| `u_stat` | Mann-Whitney U statistic for group A. |
| `p_value` | P-value for the selected alternative hypothesis. |
| `z_score` | Continuity-corrected z-score of the U used for the selected alternative's p-value. This is group A's U for `greater`, the opposite U (`n1 * n2 - u_stat`) for `less`, and the larger of the two for `two-sided`, so it corresponds to `p_value` rather than necessarily to `u_stat`. |
| `n1` | Number of observations in group A. |
| `n2` | Number of observations in group B. |

## Validation against state-of-the-art implementation

Standalone tests compare the result with `scipy.stats.mannwhitneyu`:

```python
scipy.stats.mannwhitneyu(
    group_a, group_b,
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

## Limitations and assumptions

- The p-value is approximate due to histogram rank binning. Accuracy degrades
  with very few bins or very small sample sizes.
- The normal approximation assumes sufficiently large samples. For very small
  groups (n < 10) the asymptotic p-value may not be reliable.
- Exactly two groups are compared. Multi-group comparisons are not supported.
- The two groups are assumed independent.
- Missing values are handled by the `missing_values_handler` preprocessing step
  before the test is applied.
