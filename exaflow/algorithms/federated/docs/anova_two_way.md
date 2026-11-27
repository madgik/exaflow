# Two-way ANOVA

Implementation: [statistics/anova_twoway.py](../statistics/anova_twoway.py).

Two-way analysis of variance tests whether a numerical outcome differs across
levels of two categorical factors. The fitted factorial model includes both main
effects and their interaction:

```text
y ~ A + B + A:B
```

The result is an ANOVA table with rows for the first factor, second factor,
interaction, and residuals.

## Library interface

`FederatedAnovaTwoWay(agg_client, sstype=2).fit(data=..., y=..., x1=..., x2=..., levels_a=..., levels_b=...)` accepts a dataframe and the same ordered factor levels at every participant. Constructor `sstype` is keyword-only; a fit-time `sstype` can override it. `fit` returns the estimator with `terms_`, `sum_sq_`, `df_`, `f_stat_`, and `f_pvalue_` arrays.

## Statistical method

The algorithm fits treatment-coded OLS models with an intercept. For factors
`A` and `B`, the full model is:

```text
y = beta_0 + beta_A A + beta_B B + beta_AB A:B + epsilon
```

Nested OLS models are fitted for:

| Model | Terms |
|---|---|
| `const` | Intercept only. |
| `a` | Intercept and first factor. |
| `b` | Intercept and second factor. |
| `ab` | Intercept and both main effects. |
| `full` | Intercept, both main effects, and interaction. |

Sums of squares are computed from residual sums of squares of these nested
models.

## Federated computation

The algorithm is computed without sharing row-level data. Each site builds
treatment-coded design matrices using the same factor-level order, and each OLS
fit is solved from aggregated sufficient statistics.

### Aggregated quantities

| Quantity | Purpose |
|---|---|
| Caller-supplied levels of factor `A` | Align treatment coding across sites. |
| Caller-supplied levels of factor `B` | Align treatment coding across sites. |
| `X'X` for each nested model | Estimate OLS coefficients and model rank. |
| `X'y` for each nested model | Estimate OLS coefficients. |
| Observation count | Compute residual degrees of freedom. |
| Outcome mean and centered sums of squares | Compute total variation through OLS. |
| Residual sums of squares | Compute ANOVA sums of squares and F tests. |

### Federated flow

```text
Input:
    y: numerical outcome
    A, B: categorical factors
    sstype: 1 or 2

Step 1:
    Receive the same ordered levels_a and levels_b from the caller.

Step 2:
    Validate that both factors have at least two levels.

Step 3:
    At each site:
        drop rows with missing y, A, or B
        encode A and B with the global level order
        build treatment-coded design matrices:
            const
            A-only
            B-only
            A + B
            A + B + A:B

Step 4:
    For each nested model:
        aggregate OLS sufficient statistics
        estimate the model from aggregated statistics
        store residual sum of squares, rank, and observation count

Step 5:
    Compute sums of squares:
        Type I:
            SS_A = RSS_const - RSS_A
            SS_B = RSS_A - RSS_AB
        Type II:
            SS_A = RSS_B - RSS_AB
            SS_B = RSS_A - RSS_AB
        Interaction:
            SS_A:B = RSS_AB - RSS_full
        Residual:
            SS_residual = RSS_full

Step 6:
    Compute degrees of freedom, mean squares, F statistics, and p-values.

Output:
    ANOVA table
```

## Numerical decisions

- Factor levels are globally aligned before design matrices are built.
- Treatment coding is used; the first level of each factor is the reference.
- Type II sums of squares are the default.
- Type I and Type II sums of squares are supported; Type III is not exposed.
- Tiny negative sums of squares from floating-point round-off are clipped to
  zero.
- OLS fits use pseudo-inverse based sufficient-statistic estimation.

## Reference comparison

Standalone tests compare results with `statsmodels`:

```text
lm = ols("y ~ A * B", data=df).fit()
aov = sm.stats.anova_lm(lm, typ=sstype)
```

Reference behavior is aligned with `statsmodels` OLS two-way ANOVA for Type I
and Type II sums of squares.

## Limitations

- The outcome must be numerical.
- Exactly two categorical factors are required.
- Each factor must have at least two observed levels.
- Observations are assumed independent.
- The model assumes approximately normal residuals within cells.
- The model assumes comparable residual variance across cells.
- Type III sums of squares and post-hoc comparisons are not included.
