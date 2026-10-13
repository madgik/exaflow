# Paired t-test (`feta_ttest_paired`)

## Overview

Compare two related numerical measurements using Student's paired t-test.
This algorithm preserves the response model and inference formulas of
`ttest_paired`, using independent FETA tensor execution.

## Inputs

Select one numerical variable in each required group: `y` (Measurement 1) and
`x` (Measurement 2). Both groups are prepared from one worker-local query and
refer to the same selected rows. The algorithm receives named `FedTensor` inputs.

Parameters are `alpha` (significance level, default 0.05, range 0 to 1) and
`alt_hypothesis` (`two-sided`, `less`, or `greater`, default `two-sided`). The
specification uses the existing paired algorithm's input and parameter definitions.

Differences are **x minus y**, preserving the current runtime's direction.
Missing values are rejected; preprocessing is unsupported. Workers below the
existing privacy threshold are skipped at initialization, and at least two
paired observations are required globally. Subsequent operations cannot mix
execution contexts or unrelated row partitions.

## Statistical Method

For each row let `d = x - y`. Aggregate count `n`, sample sums `sx`, `sy`,
difference sum `sd`, and squared difference sum `qd`:

```text
mean_x = sx / n
mean_y = sy / n
mean_diff = sd / n
std_diff = sqrt((qd - sd**2 / n) / (n - 1))
se_diff = std_diff / sqrt(n)
t_stat = (mean_x - mean_y) / se_diff
df = n - 1
cohens_d = mean_diff / std_diff
```

The t statistic retains the original difference-of-means numerator; the reported
mean difference and interval center use the mean of local paired differences.
Student's t distribution supplies the p-values and confidence intervals. For
`greater`, `ci_upper` is the string `Infinity`; for `less`, `ci_lower` is
`-Infinity`. Constant differences retain existing NaN/infinity behavior.

## Computation Without Row-level Data

Only aggregated sufficient statistics leave worker context storage:

| Quantity | Tensor expression | Global result |
| --- | --- | --- |
| Paired count | `len(x)` | Integer |
| Sample sums | `fnp.sum(x)`, `fnp.sum(y)` | Scalars |
| Difference sum | `fnp.sum(d)` | Scalar |
| Squared difference sum | `fnp.sum(d * d)` | Scalar |

`fnp` is `exaflow.algorithms.feta.federated_numpy`. The stored `d` tensor is reused.

### Federated Flow

```text
Prepare aligned x/y tensors; skip insufficient workers during initialization.
Compute and store local d = x - y after validating context/partition alignment.
Aggregate paired count, x sum, y sum, difference sum and squared difference sum.
Compute t statistic, p-value, interval and effect size on the controller.
Return the existing response model and clean up worker/aggregation contexts.
```

## Technical Decisions

Five successive aggregations follow the existing immediate-execution policy.
There is no implicit batching. The existing core also computes each sample's
standard deviation but does not return or use it in the paired inference; those
unused sample square sums are unnecessary here. All required computations use
existing subtraction, multiplication, length and sum operations.

## Outputs

Fields are `t_stat`, `df`, `p`, `mean_diff`, `se_diff`, `ci_upper`, `ci_lower` and
`cohens_d`. Observation count and sample standard deviations are not response
fields. Undefined numerical values follow the existing Pydantic JSON behavior.

## Validation and Limitations

Simulated federations compare every output with the existing paired core and
check t statistics/p-values against SciPy `ttest_rel`, using all alternatives,
multiple significance levels, uneven/empty partitions and privacy skips. Tests
cover insufficient observations, identical and constant-difference pairs,
different contexts and unequal lengths, cleanup and specification parity.
Real Quart/DuckDB/gRPC integration checks discovery, requests and required groups.
Deployment tests compare both algorithms on identical complete-case selections.

Raw-moment subtraction and the difference-of-means numerator retain the existing
numerical sensitivity. FETA does not silently drop incomplete pairs. See
[FETA guidance](../feta/AGENTS.md) for operation and lifecycle contracts.
