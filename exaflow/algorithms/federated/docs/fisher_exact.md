# Fisher's exact test

Implementation: [statistics/fisher_exact.py](../statistics/fisher_exact.py).

Fisher's exact test evaluates association between two binary categorical
variables using the exact probability of observing a 2 by 2 contingency table
under fixed margins.

## Library interface

`FisherExact(aggregator).compute(dataset, factor, outcome, *, factor_categories=None, outcome_categories=None, dropna=False)` returns `FisherExactTestResult`, a tuple `(odds_ratio, p_value)` with `x_labels` and `y_labels` attributes. The aggregator is a `NumpyAggregator`. See [cross-tabulation](cross_tabular_table.md) for category alignment.

## Statistical method

For a 2 by 2 table:

```text
          y1   y2
x1        a    b
x2        c    d
```

the odds ratio is:

```text
OR = (a * d) / (b * c)
```

The p-value is computed from the hypergeometric distribution over tables with
the same margins.

## Federated computation

The shared [cross-tabulation](cross_tabular_table.md) component aligns category
labels and sums local cell counts. Only the final global table is tested;
per-participant test statistics are not averaged.

### Federated flow

```text
Build the global table through CrossTabTable.compute.
Require the final pruned table to have shape (2, 2).
Apply the SciPy test to the aggregated table.
Return odds ratio and exact two-sided p-value and table labels.
```

## Numerical decisions

- The exact test is applied to the aggregated 2 by 2 table.
- Category alignment is performed before aggregation.
- No continuity correction is applied by this implementation.
- The SciPy default two-sided alternative is used.
- The global nonempty table must have shape `(2, 2)`; otherwise `FedValueError(FedErrorReason.FISHER_TABLE_SHAPE, shape=...)` is raised. The ordinary SciPy two-sided test is used.

## Reference comparison

Standalone tests compare the aggregated 2 by 2 table with:

```text
scipy.stats.fisher_exact(cross_tab)
```

Reference behavior is aligned with SciPy's default Fisher exact test.

## Limitations

- Both variables must be binary.
- The test is intended for a 2 by 2 contingency table.
- The p-value conditions on fixed row and column margins.
- The odds ratio can be infinite or undefined when cells contain zeros.
