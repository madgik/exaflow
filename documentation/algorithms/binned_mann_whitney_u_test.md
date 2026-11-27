# Binned Mann-Whitney U Test

Runtime wrapper: [statistics/binned_mann_whitney_u_test.py](../../exaflow/algorithms/exareme3/statistics/binned_mann_whitney_u_test.py).

Its federated core remains in the `statistics` family.

## Overview

The Binned Mann-Whitney U test is a non-parametric test that assesses whether
the distribution of a numerical variable differs between two independent groups.
Unlike the independent t-test, it makes no assumption about normality and
operates on ranks rather than raw values.

This implementation is a **binned approximation**: rather than computing exact
ranks across the full dataset, it groups values into histogram bins and assigns
the average rank of each bin to all values inside it. This makes exact rank
computation unnecessary — no site ever shares individual values — while
retaining the test's non-parametric properties.

The U statistic and p-value are derived from the normal approximation, matching
the approach of `scipy.stats.mannwhitneyu` with `method='asymptotic'`.

## Inputs

### Required inputs

| Input | Description |
|---|---|
| `y` | Numerical outcome variable (`REAL` or `INT`). |
| `x` | Categorical grouping variable. |

### Parameters

| Parameter | Description | Default |
|---|---|---|
| `groupA` | First group category. | Required |
| `groupB` | Second group category. | Required |
| `alt_hypothesis` | Alternative hypothesis: `two-sided`, `less`, or `greater`. | `two-sided` |
| `num_bins` | Number of histogram bins used to approximate ranks. Higher values improve accuracy. | `40` |

## Method

The mathematical method, library API, aggregation flow, numerical decisions, reference comparisons, and limitations are documented in [Binned Mann–Whitney U test](../../exaflow/algorithms/federated/docs/binned_mann_whitney_u_test.md).

## Application integration

The wrapper selects the two requested groups and supplies their numerical samples. Required missing-value preprocessing runs first; the library also removes nonfinite values. The wrapper enables continuity correction.

## Outputs

| Field | Description |
|---|---|
| `u_stat` | Mann-Whitney U statistic for group A. |
| `p_value` | P-value for the selected alternative hypothesis. |
| `z_score` | Continuity-corrected z-score of the U used for the selected alternative's p-value. This is group A's U for `greater`, the opposite U (`n1 * n2 - u_stat`) for `less`, and the larger of the two for `two-sided`, so it corresponds to `p_value` rather than necessarily to `u_stat`. |
| `n1` | Number of observations in group A. |
| `n2` | Number of observations in group B. |
