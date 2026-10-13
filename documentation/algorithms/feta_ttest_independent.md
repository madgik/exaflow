# Independent t-test (`feta_ttest_independent`)

## Overview

Compare a numerical outcome between two independent groups using the
pooled-variance Student t-test. FETA owns its specification and response model,
initially copied from `ttest_independent`; execution is independent.

## Inputs

Select one text grouping variable as required `x` and one numerical outcome as
required `y`. `groupA` and `groupB` select categories from the grouping variable.
`alpha` is the significance level (default 0.05, range 0–1).
`alt_hypothesis` is `two-sided`, `less`, or `greater` (default `two-sided`).

Worker-local string labels are preserved as NumPy Unicode arrays. Numerical
inputs remain float64. Missing values are rejected and preprocessing is
unsupported. Workers below the existing minimum-row threshold are skipped at
initialization. Subsequent filtering does not change the participating worker
set; a worker with no rows for a selected group contributes zero summaries.
Other categories are excluded from this calculation.

## Statistical Method

For each group aggregate count `n`, sum `s`, and squared sum `q`:

```text
mean = s / n
variance = (q - 2 * mean * s + mean**2 * n) / (n - 1)
df = n_a + n_b - 2
pooled_variance = ((n_a - 1) * variance_a + (n_b - 1) * variance_b) / df
se_diff = sqrt(pooled_variance * (1 / n_a + 1 / n_b))
mean_diff = mean_a - mean_b
t_stat = mean_diff / se_diff
cohens_d = mean_diff / sqrt(pooled_variance)
```

Student's t distribution supplies p-values and confidence intervals. The
one-sided upper/lower bounds use `Infinity`/`-Infinity` string markers,
respectively. Preserve the original singleton-group and constant-sample
NaN/infinity behavior. Empty global groups and total counts at most two produce
the existing errors.

## Computation Without Row-level Data

Only numerical partial counts, sums and squared sums are submitted for
aggregation. Labels, masks and selected outcome arrays stay in worker context
storage. The controller receives references, metadata and global summaries.

### Federated Flow

```python
# Inputs are prepared, aligned FedTensor objects.
grouping = x[:, 0]
a = y[grouping == groupA]
b = y[grouping == groupB]
n_a, n_b = len(a), len(b)
s_a, s_b = fnp.sum(a), fnp.sum(b)
q_a, q_b = fnp.sum(a * a), fnp.sum(b * b)
# Compute inference from these concrete summaries on the controller.
```

## Technical Decisions

Six immediate aggregation steps replace the old core's two packed calls.
Each local operation is an allowlisted request; comparisons return boolean
FedTensor values, and boolean indexing returns filtered FedTensor values.
Applying the same mask object to multiple tensors preserves their alignment;
independently created masks have separate selection lineage, even when their
values happen to match. No local array retrieval is introduced.

## Outputs

`t_stat`, `df`, `p`, `mean_diff`, `se_diff`, `ci_upper`, `ci_lower`, and `cohens_d`,
matching the existing independent t-test response.

## Validation

Federation simulations compare every field with the existing statistical core
and check t-statistics/p-values against `scipy.stats.ttest_ind(equal_var=True)`.
Coverage includes uneven and empty partitions, locally absent categories,
Unicode labels, privacy skips, alternatives, alpha values, small/constant
samples, invalid filtering, memory bounds and cleanup.

Real Quart/DuckDB/gRPC integration tests exercise `/analysis`. Five live
three-worker development-deployment cases compare both algorithms through the
same API with identical complete-case filters, using `rtol=1e-10`, `atol=1e-12`.

## Limitations

The test assumes independent observations and equal group variances. String
support is initially limited to labels, equality, transpose, column selection,
and boolean row filtering; string arithmetic and string-array literals are
unsupported. Input groups cannot mix text and numerical columns. The existing
privacy threshold is enforced during initialization; no additional per-group
release threshold is introduced by this migration.
