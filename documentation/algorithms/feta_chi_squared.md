# Chi-squared test (`feta_chi_squared`)

## Overview and inputs

Tests independence of two nominal text variables: one factor in `x` and one
outcome in `y`. Both arrive as typed `FedColumns` collections, exposing the
stored tensor and declared category codes. Missing values are rejected;
preprocessing is not supported. Workers below the existing privacy threshold
are excluded during initialization.

## Statistical method

The contingency table has factor categories as rows and outcome categories as
columns. Declared metadata order is preserved; duplicate codes are removed and
unobserved rows/columns are pruned. Without declared codes, sorted observed
labels are obtained through global category union. Observations outside a
nonempty declared domain are ignored, matching the existing algorithm.

SciPy `chi2_contingency` calculates expected counts, Pearson's chi-squared
statistic, degrees of freedom and the p-value. Its default continuity correction
is retained for one degree of freedom. A single surviving category is supported;
an empty table produces a clear input error.

## Computation without row-level data

Each worker creates float64 indicator columns using equality and multiplication
by `1.0`. Transposing the factor design and multiplying by the outcome design
produces worker-local counts. The aggregation server sums these contributions.
Only the global contingency table and, when necessary, distinct category labels
reach the controller.

### Federated flow

```text
prepare aligned x/y columns and category metadata
resolve declared categories, or globally union observed text labels
factor_design = column_stack([(factor == code) * 1.0 for code in factor_codes])
outcome_design = column_stack([(outcome == code) * 1.0 for code in outcome_codes])
table = factor_design.T @ outcome_design
prune all-zero rows and columns
calculate chi2_contingency(table) on the controller
clean up all execution contexts
```

## Technical decisions and outputs

Uses existing tensor equality, arithmetic, transpose, column stacking and
automatic matrix-product aggregation. No new worker operation or protobuf
contract is introduced. Boolean indicators are converted to floats because
NumPy boolean matrix multiplication does not calculate observation counts.

Outputs are `chi2`, `p_value`, `dof`, `expected`, `x_labels`, and `y_labels`.
Labels are category codes converted to strings, rather than display labels.

## Validation and limitations

Partitioned simulation tests compare complete results with the existing
algorithm and SciPy, covering metadata ordering, missing categories, Unicode,
privacy-based worker exclusion, continuity correction, single-level tables and
failure cleanup. Package tests exercise Quart, DuckDB, protobuf, real gRPC
workers and aggregation. Development-deployment tests compare full responses
with `chi_squared` using the existing validation cases and identical non-null
filters.

Dense indicator designs require memory proportional to observations times
category count and remain subject to worker tensor/context resource limits.
Count accumulation uses float64; ordinary dataset counts are represented
exactly up to its integer precision limit. Approximation concerns for sparse
expected counts are the same as those of the reference chi-squared test.
