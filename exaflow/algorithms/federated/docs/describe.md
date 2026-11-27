# Descriptive summaries

Implementation: [statistics/describe.py](../statistics/describe.py).

Numerical and nominal summaries are computed per caller-defined group and across all participants.

## Library interface

`FederatedDescribe(agg_client).describe(data=..., numerical_vars=..., nominal_vars=..., min_row_count=..., nominal_levels=..., group_column=..., integer_vars=None)` returns `DescribeResult(recs, global_recs)`. Callers supply the grouping column, category levels, and reporting threshold. Records identify a `variable` and `group`; the combined records have `group=None`. Report payloads contain counts and numerical summaries or category counts; suppressed payloads are `None`.

## Statistical method

For numerical variables, the algorithm reports counts, missing-value counts,
mean, sample standard deviation, minimum, quartiles, median, and maximum where
available.

For nominal variables, it reports counts, missing-value counts, total rows, and
category counts.

## Federated computation

The summaries are computed without sharing row-level data. Group-level
summaries are computed at each participant subject to a minimum row-count threshold.
Combined summaries are computed from aggregated sufficient statistics and
category counts.

### Aggregated quantities

| Quantity | Purpose |
|---|---|
| Non-missing counts | Report valid observations and compute means. |
| Missing counts | Report missingness. |
| Total row counts | Report denominator per variable and group. |
| Numerical weighted means and centered moments | Compute combined means and sample standard deviations. |
| Numerical minima and maxima | Compute combined ranges. |
| Numerical federated histogram | Estimate the combined median across all groups. |
| Nominal category counts | Compute combined category frequencies. |

### Federated flow

```text
Receive data, an explicit grouping column, and numerical/nominal variables.
For each local group and variable:
    compute missing/nonmissing counts and the requested local summaries
    suppress report payloads below the supplied row threshold
    retain weighted means and centered moments for global aggregation
Aggregate counts, weighted means, centered moments, ranges, and category counts.
Estimate combined q1, q2, q3 with the shared Percentile component.
Return group records and combined records with group=None.
```

## Numerical decisions

- Group-level records below the minimum row-count threshold return `null`
  summary data.
- Per-group numerical quartiles (`q1`/`q2`/`q3`) are exact (pandas). The
  combined summary carries federated `q1`/`q2`/`q3` and `median` (= `q2`)
  estimated across all groups from a histogram-based percentile.
- Combined numerical standard deviations use per-group centered sums of
  squares adjusted for differences between group means and the global mean.
  Divide the pooled sum by `n - 1` and take the square root.
- Nominal combined summaries use caller-supplied category levels and omit
  zero-count levels.
- Missing values are handled independently for each variable.
- Combined quartiles use the shared [Percentile](quartiles.md) component; no observation-level rows are combined.

## Reference comparison

Standalone tests compare numerical summary behavior with
`statsmodels.stats.weightstats.DescrStatsW` and verify category/count aggregation
against expected fixtures.

Reference behavior is aligned with standard descriptive-statistics formulas,
with combined summaries computed from aggregated sufficient statistics.

## Limitations

- This is descriptive only; no hypothesis tests are performed.
- Combined `q1`/`q2`/`q3`/`median` are histogram-based estimates (not exact
  quantiles); per-group quartiles are exact.
- Suppressed group-level summaries are represented with `null` data.
- Category counts depend on available caller-supplied category levels for the combined summary.
