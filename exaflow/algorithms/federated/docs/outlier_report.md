# Outlier bounds and reporting

Implementation: [outliers/outlier_report.py](../outliers/outlier_report.py).

Per-group screening computes bounds and outlier counts without pooling observation-level data.

## Library interface

`FederatedOutlierReport().report(data=..., rules=..., min_row_count=..., group_column=...)` returns per-group reports. `WinsorizationHelper.make_rules(strategies=..., tails=None, folds=None)` builds the rules; `compute_bounds(series, rule)` returns `OutlierBounds`, and `count_outliers(series, bounds)` returns lower/upper counts. Callers supply dataframe columns, the grouping column, strategies/folds, tail selection, and `min_row_count`. The helper returns bounds and counts; it does not apply clipping to the dataframe. Reports include per-variable/group bounds, tail counts, total counts, and percentages subject to suppression.

## Statistical method

For each variable and group, count values strictly outside the selected bounds. `left`, `right`, and `both` select the active tails.

| Strategy | Default fold | Bounds |
| --- | ---: | --- |
| `gaussian` | `3.0` | mean ± fold × sample standard deviation |
| `iqr` | `1.5` | Q1 − fold × IQR, Q3 + fold × IQR |
| `mad` | `3.0` | median ± fold × 1.4826 × MAD |
| `quantile` | `0.05` | fold and 1 − fold quantiles |

## Federated computation

Bounds and counts are computed locally per caller-defined group. No cross-participant moments or quantiles are aggregated.

### Federated flow

```text
Receive data, a grouping column, and per-variable outlier rules.
Resolve missing folds using the strategy defaults.
For each group and configured variable:
    drop missing/non-numeric values and check the supplied row threshold
    compute local Gaussian, IQR, MAD, or quantile bounds
    count values strictly outside the configured tails
    suppress small nonzero counts and dependent totals/percentages
Return report records; no cross-participant numerical aggregation is performed.
```

## Numerical decisions

- Bounds are group-level, not combined across all groups.
- Quantile bounds are group-level quantiles.
- `gaussian`, `iqr`, and `mad` folds must be positive finite numbers.
- `quantile` folds must be in `(0, 0.5)`.
- A zero outlier count is reported as `0`; a non-zero count below the minimum
  row-count threshold is reported as `None`.
- If either side is suppressed, total outlier count and percentage are also
  suppressed.
- Groups below `min_row_count` raise `FedObservationError` with `OUTLIER_OBSERVATIONS_INSUFFICIENT`. Consumers decide how to present this diagnostic.

## Reference comparison

The rules are aligned with standard Gaussian, IQR, MAD, and quantile screening
methods. Tests validate parameter handling, privacy masking, and expected report
fixtures rather than comparing to a single external package.

## Limitations

- Only numerical variables are supported.
- Bounds are computed per group and can differ across groups.
- Quantile, IQR, and MAD rules are descriptive screening methods, not formal
  hypothesis tests.
- Small non-zero outlier counts can be suppressed.
