# One-way ANOVA

## Table of contents

- [Overview](#overview)
- [Inputs](#inputs)
  - [Required inputs](#required-inputs)
  - [Parameters](#parameters)
- [Statistical model](#statistical-model)
- [Federated computation](#federated-computation)
  - [Aggregated quantities](#aggregated-quantities)
  - [Federated flow](#federated-flow)
- [Technical decisions](#technical-decisions)
- [Outputs](#outputs)
- [Validation against state-of-the-art implementation](#validation-against-state-of-the-art-implementation)
- [Limitations and assumptions](#limitations-and-assumptions)

## Overview

One-way analysis of variance tests whether a numerical outcome has the same mean
across the observed levels of one categorical grouping variable. The algorithm
returns an ANOVA table, group descriptive statistics, and Tukey-style pairwise
comparisons.

## Inputs

### Required inputs

| Input | Description |
|---|---|
| `y` | Numerical outcome variable. |
| `x` | One categorical grouping variable. |

### Parameters

No user parameters are exposed for this algorithm.

## Statistical model

For groups `g = 1, ..., k`, the model is:

```text
y_ij = mu + alpha_g + epsilon_ij
```

The null hypothesis is that all group means are equal:

```text
H0: mean_1 = mean_2 = ... = mean_k
```

The F statistic compares between-group variation with within-group variation:

```text
F = MS_between / MS_within
```

where `MS_between = SS_between / (k - 1)` and
`MS_within = SS_within / (n - k)`.

## Federated computation

The algorithm is computed without sharing row-level data. Each site contributes
group-level sufficient statistics, and the ANOVA table is derived from the
aggregated counts, sums, and sums of squares.

### Aggregated quantities

| Quantity | Purpose |
|---|---|
| Group counts | Define group sizes and degrees of freedom. |
| Group sums | Compute group means and between-group sums of squares. |
| Group sums of squares | Compute within-group sums of squares and sample standard deviations. |
| Overall count, sum, and sum of squares | Compute total sample size and the overall mean. |
| Group minima and maxima | Report group-level descriptive ranges. |

### Federated flow

```text
Input:
    y: numerical outcome
    x: categorical grouping variable

Step 1:
    Determine the globally observed group labels.

Step 2:
    At each site, for each group:
        remove missing values
        compute count
        compute sum(y)
        compute sum(y^2)
        compute minimum and maximum

Step 3:
    Aggregate group counts, sums, sums of squares, minima, and maxima.

Step 4:
    Remove globally empty groups.
    Validate that at least two groups remain.

Step 5:
    Compute:
        overall mean
        between-group sum of squares
        within-group sum of squares
        mean squares
        F statistic
        p-value

Step 6:
    Compute pairwise Tukey-style comparisons from aggregated group means,
    residual mean square, and group counts.

Output:
    ANOVA table, group summaries, and pairwise comparisons
```

## Technical decisions

- Empty groups are removed after aggregation.
- At least two non-empty groups are required.
- Within-group variance is computed from aggregated sums and sums of squares.
- The pairwise comparison table uses the residual mean square from the fitted
  one-way ANOVA model.
- Missing values are removed before group statistics are computed.

## Outputs

| Field | Description |
|---|---|
| `n_obs` | Number of observations used in the test. |
| `f_stat` | ANOVA F statistic. |
| `p_value` | P-value for the omnibus group-mean test. |
| `df_between` | Between-group degrees of freedom. |
| `df_within` | Residual degrees of freedom. |
| `ss_between` | Between-group sum of squares. |
| `ss_within` | Within-group sum of squares. |
| `ms_between` | Between-group mean square. |
| `ms_within` | Within-group mean square. |
| `group_stats` | Counts, means, standard deviations, minima, and maxima by group. |
| `tukey_hsd` | Pairwise group comparisons. |

## Validation against state-of-the-art implementation

Standalone tests compare the ANOVA table with `statsmodels` OLS ANOVA using a
centralized reference model:

```text
ols("y ~ x", data=df).fit()
sm.stats.anova_lm(lm)
```

Pairwise comparison behavior is aligned with the Tukey HSD methodology exposed
through `statsmodels.stats.libqsturng.psturng`.

## Limitations and assumptions

- The outcome must be numerical.
- The grouping variable must have at least two non-empty levels.
- Observations are assumed independent.
- The model assumes approximately normal residuals within groups.
- The model assumes comparable within-group variances.
- The pairwise comparison table is post-hoc and uses the fitted ANOVA residual
  mean square.
