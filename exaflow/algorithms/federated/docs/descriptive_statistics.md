# Descriptive statistics facade

Implementation: [statistics/descriptive_stats.py](../statistics/descriptive_stats.py).

`FederatedDescriptiveStatistics(agg_client)` groups three existing components:

- `describe(...)` delegates to [descriptive summaries](describe.md).
- `hist(...)` delegates to [grouped histograms](histogram.md).
- `corrcoef(...)` and `pearson_correlation(...)` delegate to
  [Pearson correlation](pearson_correlation.md).

Inputs, defaults, results, aggregation operations, validation, and numerical
limitations are those of the delegated method. The facade adds no statistical
estimation or new aggregation. Callers supply the grouping column, category
levels/metadata, and minimum row count where the method requires them.
