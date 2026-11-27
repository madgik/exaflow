# Two-way ANOVA

## Overview

Two-way analysis of variance tests whether a numerical outcome differs across
levels of two categorical factors. The fitted factorial model includes both main
effects and their interaction:

```text
y ~ A + B + A:B
```

The result is an ANOVA table with rows for the first factor, second factor,
interaction, and residuals.

## Inputs

### Required inputs

| Input | Description |
|---|---|
| `y` | Numerical outcome variable. |
| `x` | Exactly two categorical factors. |

### Parameters

| Parameter | Description | Default |
|---|---|---|
| `sstype` | Sums-of-squares decomposition used for the ANOVA table. | `2` |

Supported values:

| Value | Method |
|---|---|
| `1` | Type I sequential sums of squares. |
| `2` | Type II marginal sums of squares. |

## Method

The mathematical method, library API, aggregation flow, numerical decisions, reference comparisons, and limitations are documented in [Two-way ANOVA](../../exaflow/algorithms/federated/docs/anova_two_way.md).

## Outputs

| Field | Description |
|---|---|
| `terms` | ANOVA terms. |
| `sum_sq` | Sum of squares for each term. |
| `df` | Degrees of freedom for each term. |
| `f_stat` | F statistic for testable terms. |
| `f_pvalue` | P-value for testable terms. |

Residual rows do not have F statistics or p-values.
