# Chi-squared test

Implementation: [statistics/chi_squared.py](../statistics/chi_squared.py).

The chi-squared test evaluates whether two categorical variables are independent.
It builds a contingency table for an outcome variable and a factor variable,
then compares observed cell counts with expected counts under independence.

## Library interface

`ChiSquared(aggregator).compute(dataset, factor, outcome, *, factor_categories, outcome_categories, dropna=False)` returns a tuple-like `ChiSquaredTestResult`: `(chi2, p_value, dof, expected)`, with `x_labels` and `y_labels` attributes. The aggregator is a `NumpyAggregator`. See [cross-tabulation](cross_tabular_table.md) for category alignment and table construction.

## Statistical method

For a contingency table with observed counts `O_ij`, expected counts under
independence are:

```text
E_ij = row_sum_i * column_sum_j / n
```

The test statistic is:

```text
chi2 = sum_ij (O_ij - E_ij)^2 / E_ij
```

Degrees of freedom are:

```text
(rows - 1) * (columns - 1)
```

## Federated computation

The shared [cross-tabulation](cross_tabular_table.md) component aligns category
labels and sums local cell counts. Only the final global table is tested;
per-participant test statistics are not averaged.

### Federated flow

```text
Build the global table through CrossTabTable.compute.
Apply the SciPy test to the aggregated table.
Return chi-squared statistic, p-value, degrees of freedom, and expected counts and table labels.
```

## Numerical decisions

- Category alignment is performed before count aggregation.
- Missing values may be represented as table categories when included by the
  cross-tabulation helper.
- The test uses the aggregated contingency table, not per-site test statistics.
- Expected counts are derived after aggregation.
- SciPy `chi2_contingency` defaults apply, including Yates continuity correction when the degrees of freedom equal one. The uncorrected formula above describes the Pearson statistic before that correction.

## Reference comparison

Standalone tests compare the aggregated contingency table result with:

```text
scipy.stats.chi2_contingency(cross_tab)
```

Reference behavior is aligned with SciPy's chi-squared test on the centralized
contingency table.

## Limitations

- Both variables must be categorical.
- The usual chi-squared approximation assumes sufficiently large expected cell
  counts.
- Sparse tables can make the asymptotic p-value unreliable.
- The test detects association but does not estimate effect direction or causal
  relationships.
