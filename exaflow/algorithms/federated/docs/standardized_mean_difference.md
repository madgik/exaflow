# Standardized Mean Difference

`FederatedStandardizedMeanDifference` computes Cohen's d for independent groups from aggregated sufficient statistics. Raw observations remain local; only counts, sums, and squared deviations are combined.

## Inputs

The `compute(x, y)` method accepts two one-dimensional NumPy arrays. The arrays may have different lengths. Missing numeric values are excluded by the aggregation layer, and each group must contain at least two valid observations.

The `pairwise_by_group(data=..., value_var=..., group_var=...)` method accepts a local DataFrame partition, the continuous value column, and the grouping column. It discovers the union of observed group labels and computes every unique group pair in deterministic order. Groups with fewer than two valid values are omitted from all comparisons.

## Statistical method

For groups 1 and 2, the method calculates sample variances with one degree of freedom and the pooled variance

```text
s_p^2 = ((n1 - 1) var1 + (n2 - 1) var2) / (n1 + n2 - 2)
```

Cohen's d is then

```text
d = (mean1 - mean2) / sqrt(s_p^2)
```

If the pooled variance is zero, the result is `0.0`.

Both public operations call the same `_smd(n1, mean1, var1, n2, mean2, var2)` helper after aggregation, so the effect-size calculation is identical for direct and grouped comparisons.

## Federated flow

```text
Discover all group labels across data holders.
For each group, aggregate its valid count and mean.
Use the global mean to aggregate squared deviations and calculate sample variance.
Generate each unique pair of eligible groups.
Calculate Cohen's d from the two groups' summary statistics.
```

## Outputs

`compute` returns one floating-point Cohen's d value.

`pairwise_by_group` returns a DataFrame with `group1`, `group2`, `n1`, `n2`, `mean1`, `mean2`, `var1`, `var2`, and `smd`. An empty result preserves these columns.

## Technical decisions and limitations

The method implements the equal-variance form of Cohen's d. It does not apply Hedges' small-sample correction and does not calculate confidence intervals or hypothesis-test p-values. Pair orientation determines the sign: each result is `group1 - group2`.

Validation compares results with NumPy and pandas calculations using sample variance (`ddof=1`) for one and multiple data partitions, including disjoint groups, uneven partitions, missing values, numeric group labels, and insufficient groups.
