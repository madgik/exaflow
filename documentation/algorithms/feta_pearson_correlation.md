# Pearson Correlation (`feta_pearson_correlation`)

## Overview

Compute Pearson coefficients, two-sided p-values, and Fisher-transformed
confidence intervals between all pairs of selected numerical variables.
The FETA algorithm is separate from `pearson_correlation` and preserves that
algorithm's output contract and statistical formulas.

## Inputs

Select a data model, datasets, source variables, and optional filters through
the standard analysis request. `y` is the required primary variable group;
`x` is an optional secondary group. Omitted `x` resolves to `y`. Both groups
preserve the requested variable order and use the same selected rows.

`alpha` is required by the public specification and defaults to `0.95` in its
metadata. The existing implementation labels it a confidence level but uses it
as a significance level in `norm.ppf(1 - alpha / 2)`. This implementation
preserves that behavior; use `alpha=0.05` for the usual 95% Fisher interval.

Inputs are numerical `float64` matrices. Missing values cause an input error;
preprocessing, validation datasets, and optional execution flags are unsupported.
Workers below the existing minimum-row threshold are skipped before computation.

## Statistical Method

For every pair `(x_i, y_j)`, collect observation count `n`, sums `s_x`, `s_y`,
squared sums `s_xx`, `s_yy`, and cross-products `s_xy`. The coefficient is

```text
r_ji = (n * s_xy_ji - s_x_i * s_y_j)
       / sqrt((n * s_xx_i - s_x_i**2) * (n * s_yy_j - s_y_j**2))
```

Zero-denominator coefficients are set to zero and all coefficients are clipped
to `[-1, 1]`. With `df = n - 2`, the p-value uses the incomplete-beta expression
for the two-sided Pearson t statistic. Perfect correlations have p-value zero.

Intervals use `arctanh(r)`, standard error `1 / sqrt(n - 3)`, and
`norm.ppf(1 - alpha / 2)`, followed by `tanh`. Empty data and `n <= 2` cause an
input error. Three observations are accepted with the existing singular
interval behavior, rather than introducing a different minimum sample size.

## Computation Without Row-Level Data

Initial rows and intermediates remain in worker-local context storage.
Only partial sufficient statistics are submitted to the aggregation server;
the controller receives their global sums and computes the statistical outputs.

### Aggregated Quantities

| Quantity | Tensor expression | Global shape |
| --- | --- | --- |
| Observation count | `len(x)` | Integer |
| Feature sums | `fnp.sum(x, axis=0)`, `fnp.sum(y, axis=0)` | Feature vectors |
| Squared sums | `fnp.sum(x * x, axis=0)`, `fnp.sum(y * y, axis=0)` | Feature vectors |
| Cross-products | `fnp.transpose(y) @ x` | `(y_features, x_features)` |

Here `fnp` is `exaflow.algorithms.feta.federated_numpy`, imported with
`from exaflow.algorithms.feta import federated_numpy as fnp`. These typed
functions use the same execution path as the tensor's `.sum()` and `.T`.

### Federated Flow

```text
Resolve ordered x/y columns and initialize local tensors from one query.
Skip workers with insufficient rows; fix the remaining participant set.
For each of the six global sufficient-statistic expressions:
    Controller prepares the aggregation context and step.
    Workers execute approved local operations and submit their partials.
    Controller fetches the completed sum directly from the aggregation server.
Compute coefficients, p-values, and intervals using NumPy/SciPy.
Format the existing response dictionaries and clean up all contexts.
```

## Technical Decisions

The numerical cross-product is `fnp.transpose(y) @ x`, equivalent to `y.T @ x`
and equivalent in orientation to the
original `np.einsum("ji,jk->ki", x, y)`. The runtime determines aggregation
from partitioning, rather than requiring an explicit global-sum call.
Column selection and row alignment happen before `run(x, y)` receives tensors.
Any participating-worker failure aborts execution; operations are not retried.

See [FETA architecture](../feta/AGENTS.md) for operation validation, storage,
limits, expiry, protocol regeneration, and deployment compatibility.

## Outputs

The returned Pydantic model has `title`, `n_obs`, `correlations`, `p_values`,
`ci_hi`, and `ci_lo`. Every matrix dictionary has `variables` containing ordered
`x` names, and one key for each ordered `y` name holding its vector across `x`.
The title is `Pearson Correlation Coefficient`. JSON serialization follows the
existing Pearson result model, including undefined numerical results.

## Validation

Standalone simulations compare every output against the existing Pearson core
with sufficient-statistic aggregation, using one worker and uneven partitions.
Package tests run the real `/analysis` route with DuckDB and localhost gRPC
workers/aggregation. Tests also cover omitted `x`, rectangular orientation,
constant features, observation edge cases, filtering, privacy skips, errors,
and context cleanup. No production deployment is required for those tests.

The p-value method is aligned with `scipy.stats.pearsonr`; the interval formula
is consistent with the Fisher transformation. Exact equivalence is not claimed
for constant inputs or small-sample interval handling.

## Limitations

Sufficient-statistic subtraction can lose precision for nearly constant or
large-offset variables. Aggregation order can introduce small floating-point
differences. Missing values are rejected, and a context that reaches the
configured maximum lifetime fails even if computation remains active.
