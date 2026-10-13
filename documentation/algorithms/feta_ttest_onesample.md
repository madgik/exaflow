# One-sample t-test (`feta_ttest_onesample`)

## Overview

Compare a numerical sample mean with a reference mean using Student's t-test.
This algorithm preserves the output model and statistical formulas of
`ttest_onesample`, with an independent tensor execution flow.

## Inputs

Select one numerical variable in `y`, a data model, datasets and optional filters.
`x` is not a public input. The runtime prepares only `y`, and invokes
`run(y=y)` with its worker-local tensor references.

Parameters are `mu` (reference mean, default 0, range -10 to 10), `alpha`
(significance level, default 0.05, range 0 to 1), and `alt_hypothesis`
(`two-sided`, `less`, or `greater`, default `two-sided`). Public specification
validation uses the same parameter definitions as the existing algorithm.

Missing values are rejected and preprocessing is unsupported. Workers below
the existing minimum row count are skipped during initialization. If none
qualify, execution fails. At least two observations are required globally.

## Statistical Method

For global count `n`, sum `s` and squared sum `q`, compute:

```text
d = s - n * mu
q_d = q - 2 * mu * s + n * mu**2
mean = s / n
std = sqrt((q_d - d**2 / n) / (n - 1))
se = std / sqrt(n)
t = (mean - mu) / se
df = n - 1
```

Student's t distribution supplies the p-value and confidence intervals. Cohen's
d is `(mean - mu) / std`. The confidence interval is centered on the sample mean.
For `greater`, `ci_upper` is the string `Infinity`; for `less`, `ci_lower` is
`-Infinity`. Constant-sample NaN/infinity behavior follows the existing formulas.

## Computation Without Row-level Data

Only three aggregated quantities leave the workers:

| Quantity | Tensor expression | Global result |
| --- | --- | --- |
| Count | `len(y)` | Integer |
| Sum | `fnp.sum(y)` | Scalar |
| Squared sum | `fnp.sum(y * y)` | Scalar |

`fnp` refers to `exaflow.algorithms.feta.federated_numpy`. Sample arrays and
intermediates stay in worker context storage.

### Federated Flow

```text
Initialize aligned sample tensors and fix the eligible workers.
Aggregate length, sample sum and squared sum in three successive steps.
Derive reference-centered moments from completed sums on the controller.
Compute t statistic, tails, confidence interval and effect size.
Return the FETA-owned response model and clean up worker/aggregation contexts.
```

## Technical Decisions

The existing implementation also aggregates local reference-centered moments.
These are linear combinations of count, sum and squared sum; this version
derives them from the completed global values. Minor floating-point differences
from summation order are permitted. No additional worker operators are required.

## Outputs

Fields are `n_obs`, `std`, `t_stat`, `df`, `p`, `mean_diff`, `se_diff`, `ci_upper`,
`ci_lower`, and `cohens_d`. Despite its name, `mean_diff` preserves the existing
sample-mean output rather than subtracting `mu`. Undefined numerical values use
the copied FETA-owned Pydantic model's JSON serialization.

## Validation and Limitations

Mocked federation simulations compare every output field with the existing core,
using uneven and empty partitions, privacy skips, all alternatives, constant
samples and small-sample errors. SciPy `ttest_1samp` checks the t statistic and
p-value. Real Quart/DuckDB/gRPC integration verifies discovery and `/analysis`.
A deployment parity test compares both runtime algorithms on identical
complete-case filtered populations.

There is no missing-value preprocessing, and raw-moment subtraction retains the
existing numerical sensitivity for nearly constant samples. See
[FETA guidance](../feta/AGENTS.md) for lifecycle and protocol contracts.
