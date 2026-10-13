# One-way ANOVA (`feta_anova_oneway`)

## Overview

Compare a numerical outcome across categories of one grouping variable using
one-way ANOVA, with Tukey HSD pairwise comparisons. The algorithm preserves the
existing `anova_oneway` response and inference formulas, while executing through
FETA tensor expressions.

## Inputs

Required `x` selects one nominal text grouping variable. Required `y` selects one
numerical outcome. There are no public parameters. Input ordering and row
alignment follow the specification and validated request.

Text labels are retained on workers; numerical inputs use float64. Missing
values are rejected and preprocessing is unsupported. Workers below the existing
privacy minimum-row threshold are skipped during initialization. Categories
occurring only on skipped workers are excluded.

## Statistical Method

Discover sorted globally observed categories. For each group, aggregate count
`n`, sum `s`, squared sum `q`, minimum and maximum. Let `N`, `S`, `Q` be totals of
the group counts, sums and squared sums, and `k` the number of observed groups:

```text
group_mean = s / n
overall_mean = S / N
ss_within = Q - sum(s**2 / n)
ss_between = sum(n * (group_mean - overall_mean)**2)
df_between = k - 1
df_within = N - k
ms_between = ss_between / df_between
ms_within = ss_within / df_within
F = ms_between / ms_within
```

The F distribution gives the group-effect p-value. Tukey comparisons use mean
differences, `sqrt(ms_within * (1/n_a + 1/n_b))`, and the existing `psturng`
calculation for the studentized range distribution.

## Computation Without Row-level Data

Distinct category labels are combined by aggregation-server union. Counts, sums,
squared sums and extrema are aggregated for each category. Individual outcome
arrays, row masks and filtered samples remain in worker context storage.

### Federated Flow

```python
grouping = x[:, 0]
categories = fnp.unique(grouping)
for category in categories:
    sample = y[grouping == category]
    n = len(sample)
    s = fnp.sum(sample)
    q = fnp.sum(sample * sample)
    minimum = fnp.min(sample)
    maximum = fnp.max(sample)
# Compute ANOVA, Tukey comparisons and output tables from these summaries.
```

One union plus five immediate reductions per category gives `1 + 5*k`
aggregation steps. No batching is introduced. Workers with empty local groups
remain participants: counts/sums contribute zero, minima contribute positive
infinity and maxima contribute negative infinity. Every discovered category has
at least one observation globally.

## Technical Decisions

`federated_numpy.unique` returns a sorted ordinary NumPy array. Distributed
unique initially supports string category labels only; it does not expose
unique numerical observations. Numeric NumPy arrays can use the function
locally. Variable-length label contributions use typed protobuf string arrays,
with dtype width and candidate-count bounds validated by the aggregation server
and controller. Bounds based on input sizes are conservative and can reject a
large request even when its final category set would be small.

`federated_numpy.min` and `max` retain a FedTensor when the reduced axis is local,
and use global min/max when reducing the partitioned axis. Globally empty
reductions raise an error; empty individual worker contributions are neutral.
NaN and infinity follow NumPy reduction behavior rather than a missing-aware
policy. Missing input values remain rejected at initialization.

FETA owns `AnovaStatistics` and its output-formatting helper, copied from the
reference implementation. It calls the owned inference after tensor aggregation.
The shared federated core and Exareme3 implementations remain unchanged;
there are no production imports from those algorithm libraries.

## Outputs

- `anova_table`: observation count, input labels, degrees of freedom, sums and
  means of squares, F-statistic and p-value.
- `tuckey_test`: pairwise group labels, means, differences, standard errors,
  t-statistics and Tukey p-values. Preserve this existing field spelling.
- `min_max_per_group`: sorted categories and their global outcome extrema.
- `ci_info`: per-group means, reported standard deviations and mean-minus/plus
  standard-deviation values.

## Validation

Simulated federation tests compare every nested field with the existing runtime
and F-statistics/p-values with `scipy.stats.f_oneway`. They cover uneven and empty
partitions, Unicode labels, privacy skips, categories present on only one worker,
insufficient levels, constant observations and cleanup.

Real Quart/DuckDB/gRPC integration tests execute `/analysis`. Development-tool
deployment tests compare all five original ANOVA fixture cases with normal
ANOVA through the API, using identical complete-case filters and tolerances
`rtol=1e-10`, `atol=1e-12`.

## Limitations and Preserved Behavior

Observations should be independent, with approximately normal residuals and
comparable within-group variances. Fewer than two observed groups produce the
existing error. Preserve zero-within-variance behavior, including the original
choice of F=0 instead of the usual infinity when residual mean square is zero.

The reported standard deviation uses the existing formula
`sqrt((n-1)/n * (q/n - mean**2))`; it is not the unbiased sample standard
deviation. The `ci_info` bounds are mean plus/minus that value, rather than
confidence intervals based on a specified confidence level. These quirks are
preserved during migration. Degenerate residual degrees of freedom and
nonfinite inference retain the existing core behavior.

Category labels and per-group extrema are released as in normal ANOVA. The
migration preserves initialization privacy thresholds and introduces no new
per-group privacy threshold or unrestricted local-array retrieval.
