# One-way ANOVA

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

## Method

The mathematical method, library API, aggregation flow, numerical decisions, reference comparisons, and limitations are documented in [One-way ANOVA](../../exaflow/algorithms/federated/docs/anova_one_way.md).

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
