# Fisher's exact test (`feta_fisher_exact`)

## Overview and inputs

Tests association between one nominal text factor in `x` and one nominal text
outcome in `y`. Inputs arrive as aligned `FedTensor` values. Each variable must
have exactly two globally observed levels among eligible workers. Missing
values are rejected; preprocessing is not supported. Initialization skips
workers below the existing privacy threshold.

## Statistical method

The 2x2 contingency table contains factor categories as rows and outcome
categories as columns. Categories are sorted observed codes, independently of
metadata enumeration order or unobserved declared categories. SciPy
`fisher_exact` computes the sample odds ratio and the default two-sided exact
p-value, matching the existing runtime algorithm. Zero cells are supported and
can produce an infinite odds ratio.

## Computation without row-level data

Distinct text labels are combined through aggregation-server union. Workers
construct float64 indicator columns and local matrix products. The aggregation
server sums the products into the global 2x2 table. Only observed distinct labels
and aggregated counts reach the controller; observation arrays remain local.

### Federated flow

```text
prepare aligned factor and outcome tensors
factor_codes = unique(factor)
outcome_codes = unique(outcome)
require exactly two observed levels in each group
factor_design = column_stack([(factor == code) * 1.0 for code in factor_codes])
outcome_design = column_stack([(outcome == code) * 1.0 for code in outcome_codes])
table = factor_design.T @ outcome_design
odds_ratio, p_value = scipy.stats.fisher_exact(table)
clean up execution contexts
```

## Technical decisions and outputs

Reuses the FETA-owned indicator-table helper and existing tensor operations;
no new local operation, transport message or deployment configuration is needed.
There are three aggregation steps for a valid execution. Invalid observed-level
counts raise the same descriptive input error as the existing algorithm.

Outputs are `odds_ratio`, `p_value`, `x_labels`, and `y_labels`. Labels contain
sorted category codes, not display labels. Degenerate results retain the
existing response model's non-finite-value serialization behavior.

## Validation and limitations

Simulation tests compare complete results with the existing implementation and
SciPy, covering multiple partitions, Unicode, zero cells, invalid level counts,
privacy-based exclusions and cleanup. Package tests cover the real analysis
route, DuckDB loading, protobuf, workers and aggregation RPCs. Development
validation compares full responses against `fisher_exact` using existing test
fixtures and identical non-null filters.

Only 2x2 tables are supported. The initial missing-value rejection policy differs
from the reference algorithm's preprocessing-based deletion; explicit non-null
filters provide identical selected observations for comparisons. Indicator
allocation is subject to the existing worker resource limits. Float64 counts
retain exact integer representation up to its integer precision limit.
