# Cross-tabulation

Implementation: [statistics/cross_tabular_table.py](../statistics/cross_tabular_table.py).

`CrossTabTable(aggregator).compute(dataset, factor, outcome, *, factor_categories=None, outcome_categories=None, dropna=False)` builds a global
contingency table from a local dataframe using `NumpyAggregator`. The result is
a dataframe indexed by factor levels, with outcome levels as columns.

## Method and aggregated quantities

Each cell counts observations with a given factor/outcome combination. Summing
aligned local tables produces the pooled table. Callers can supply ordered
category lists; otherwise categories are discovered through a union. Missing
values are retained as a separate category by default, with their presence
synchronized across participants.

### Federated flow

```text
Receive the local dataframe and factor/outcome column names.
Use supplied category lists or discover the common category universe.
Synchronize missing-category presence when dropna=False.
Construct and reindex local cross-tabs in the same row/column order.
Sum all cell counts.
Remove globally all-zero rows and columns.
Return the nonempty global dataframe.
```

## Decisions and limitations

An empty pruned table raises `FedValueError` with `EMPTY_CONTINGENCY_TABLE`.
Explicit categories absent everywhere are removed. Supplied categories must
cover the intended population; reindexing does not discover extra categories
when a nonempty explicit list is supplied. Participants must agree on category
configuration. Counts are exact for the resulting table; hypothesis-test
assumptions are owned by [chi-squared](chi_squared.md) and
[Fisher's exact test](fisher_exact.md).
