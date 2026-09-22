# Standardized Mean Difference

## Overview

Standardized Mean Difference computes Cohen's d for every unique pair of categories in one grouping variable. It reports the size and direction of each difference relative to the groups' pooled within-group variability.

## Inputs

- **Grouping variable (`x`)**: one nominal text or integer variable. Its observed categories define the comparisons.
- **Outcome (`y`)**: one numerical variable whose means are compared.

Rows with a missing group label or outcome value do not contribute. A category must reach the configured privacy minimum to appear in a comparison.

## Statistical method

For groups 1 and 2, the algorithm computes sample variances and the pooled variance

```text
s_p^2 = ((n1 - 1) var1 + (n2 - 1) var2) / (n1 + n2 - 2)
```

The reported effect size is

```text
smd = (mean1 - mean2) / sqrt(s_p^2)
```

A zero pooled variance produces `smd = 0.0`. The method implements Cohen's d without Hedges' small-sample correction.

## Computation without row-level data

Each data holder calculates local contributions for group counts, sums, and squared deviations. Only aggregated values are used to derive global means, sample variances, and pairwise effect sizes. Individual observations are not returned.

## Aggregated quantities

For every eligible group, the computation combines:

- the valid observation count;
- the sum used for the global mean;
- the sum of squared deviations used for sample variance.

### Federated flow

```text
Discover the union of observed group labels.
Aggregate each group's valid count and mean.
Aggregate squared deviations around each global group mean.
Discard groups below the configured privacy minimum.
Generate every unique pair of remaining groups.
Compute Cohen's d from the paired summary statistics.
```

## Technical decisions

Groups are ordered deterministically by their string representation. Therefore, each pair appears once, and the sign represents `group1 - group2`. Groups that fail the minimum size are omitted rather than returned with partial statistics. Missing values are excluded before counts and moments are calculated.

## Outputs

The response contains `comparisons`, a list whose entries contain:

- `group1`, `group2`: compared category labels;
- `smd`: pooled-variance Cohen's d.

When fewer than two eligible groups remain, `comparisons` is empty.

## Validation

Standalone tests compare direct and grouped results with NumPy and pandas sample-statistic calculations across one and multiple data partitions. Cases cover unequal sizes, disjoint groups, skewed partitions, missing values, numeric labels, zero variance, and insufficient groups. Production validation uses the same pooled-variance reference calculation on the dementia test model.

## Limitations

The calculation assumes the pooled-variance definition is suitable for the compared groups. It does not provide confidence intervals, hypothesis-test p-values, bias correction, or an unequal-variance effect-size variant. The number of returned comparisons grows quadratically with the number of eligible categories.
