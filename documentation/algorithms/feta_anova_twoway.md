# Two-way ANOVA (`feta_anova_twoway`)

## Overview

Test a numerical outcome against two categorical factors and their interaction.
The model is `y ~ A * B`. Type I sequential and Type II marginal sums of squares
preserve the existing `anova_twoway` response and formulas, using FETA execution.

## Inputs

Required `x` contains exactly two nominal text variables, in requested factor
order. Required `y` contains one numerical outcome. `sstype` selects 1 or 2;
the default is Type II. Type I depends on factor order.

Factors retain worker-local string labels; outcomes use float64. Missing values
are rejected and preprocessing is unsupported. Initialization skips workers below
the existing privacy threshold. Each factor must have at least two observed
levels across eligible workers; the original error messages are preserved.

## Statistical Method

Globally discovered levels are sorted. Treatment coding omits the first level
of each factor. Products of treatment indicators form the interaction columns.
Five designs model the intercept, A, B, A+B and A+B+A:B.

For each design `D`, aggregate its Gram matrix and response cross-product:

```text
G = D.T @ D
c = D.T @ y
coefficients = pinv(G) @ c
rank = matrix_rank(G)
residuals = y - D @ coefficients
RSS = sum(residuals**2)
```

Let `RSS_0`, `RSS_A`, `RSS_B`, `RSS_AB`, and `RSS_full` denote model residual sums:

| Term | Type I sum of squares | Type II sum of squares |
| --- | --- | --- |
| A | `RSS_0 - RSS_A` | `RSS_B - RSS_AB` |
| B | `RSS_A - RSS_AB` | `RSS_A - RSS_AB` |
| A:B | `RSS_AB - RSS_full` | `RSS_AB - RSS_full` |
| Residuals | `RSS_full` | `RSS_full` |

Negative rounding artifacts are clipped to zero. Main-effect degrees of freedom
are their individual model rank minus intercept-model rank; interaction degrees
of freedom are full-model rank minus additive-model rank. Residual degrees of
freedom are observation count minus full-model rank, bounded below by zero.
F-statistics divide each term's mean square by the full-model residual mean
square. The F distribution supplies p-values.

## Computation Without Row-level Data

The controller receives observed category labels, one count, model Gram matrices,
cross-products and residual squared sums. Outcome rows, indicator columns,
design matrices and predictions remain in worker context storage.

### Federated Flow

```python
a, b = x[:, 0], x[:, 1]
levels_a, levels_b = fnp.unique(a), fnp.unique(b)
intercept = fnp.ones_like(y)
main_a = [(a == level) * 1.0 for level in levels_a[1:]]
main_b = [(b == level) * 1.0 for level in levels_b[1:]]
interaction = [left * right for left in main_a for right in main_b]
# Assemble the five aligned designs with fnp.column_stack.
# For each design, aggregate D.T @ D and D.T @ y.
# Compute coefficients on the controller and broadcast them via D @ coefficients.
# Aggregate squared local residuals; derive the ANOVA table from model summaries.
```

Execution uses 18 immediate aggregation steps: two label unions, one observation
count, and three sums per model. No automatic batching is introduced. Model
summaries unused by ANOVA, such as OLS absolute residuals and likelihoods, are
not computed.

## Technical Decisions

FETA owns its specification, response model, mathematical inference and tensor
expressions. No Exareme3 or shared algorithm implementation is imported or
modified. The copied table inference is in `_anova_twoway.py`.

`fnp.ones_like` creates a stored numerical/boolean tensor with the input shape,
dtype and partitioning. `fnp.column_stack` combines aligned row-partitioned
vectors and matrices; vectors become single columns. All distributed inputs
must be FedTensors in the same context and row layout. Mixing numerical and
text columns, local and distributed inputs, or column-partitioned inputs fails.
Output allocation is checked before execution.

The normal implementation constructs an intercept and its OLS layer adds another
one. FETA retains these two identical intercept columns to preserve its Gram
matrix, rank calculation and solver rounding. Pseudoinverse handles this
redundancy and other rank deficiencies. This migration does not correct that
reference behavior.

## Outputs

`terms`, `sum_sq`, `df`, `f_stat` and `f_pvalue` are parallel lists in requested
factor, second factor, interaction and residual order. Residual F-statistic and
p-value are null. Untestable effects, zero residual degrees of freedom or zero
residual mean square retain the original null/error behavior.

## Validation

Federation simulations compare every field with normal two-way ANOVA. Full-rank
cases independently compare both sums-of-squares types with statsmodels
`ols('value ~ C(a) * C(b)')` and `anova_lm(..., typ=sstype)`.
Coverage includes uneven/empty partitions, privacy skips, Unicode labels,
missing cells, saturated designs, insufficient levels, factor order, allocation
limits and cleanup.

Package tests run the actual Quart/DuckDB/gRPC `/analysis` flow for both types.
Development deployment tests compare all original two-way ANOVA fixture cases
with the normal algorithm through the API, using identical complete-case filters
and `rtol=1e-10`, `atol=1e-12` tolerances.

## Limitations

Independence, approximate residual normality and comparable within-cell variances
are the usual assumptions. Dense designs and retained intermediate tensors are
bounded by FETA's context/tensor memory limits. Rank-deficient designs preserve
existing rank-based behavior; not every classical contrast interpretation is
available. There is no preprocessing, generic design-formula API or numerical
observation release through global unique. Privacy thresholds and lifecycle
cleanup remain unchanged.
