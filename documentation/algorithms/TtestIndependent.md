# Independent t-test

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

The independent t-test compares the mean of a numerical variable between two
independent groups. This implementation uses the pooled-variance Student t-test
and reports the t statistic, p-value, confidence interval, standard error, and
Cohen's d.

## Inputs

### Required inputs

| Input | Description |
|---|---|
| `y` | Numerical outcome variable. |
| `x` | Categorical grouping variable. |

### Parameters

| Parameter | Description | Default |
|---|---|---|
| `groupA` | First group category. | Required |
| `groupB` | Second group category. | Required |
| `alpha` | Significance level for confidence intervals. | `0.05` |
| `alt_hypothesis` | Alternative hypothesis: `two-sided`, `less`, or `greater`. | `two-sided` |

## Statistical model

The null hypothesis is:

```text
H0: mean_A = mean_B
```

With pooled variance:

```text
s_p^2 = ((n_A - 1)s_A^2 + (n_B - 1)s_B^2) / (n_A + n_B - 2)
t = (mean_A - mean_B) / sqrt(s_p^2 * (1 / n_A + 1 / n_B))
df = n_A + n_B - 2
```

## Federated computation

The test is computed without sharing row-level data. Each site contributes
group-specific counts, sums, and sums of squares for the selected categories.

### Aggregated quantities

| Quantity | Purpose |
|---|---|
| Count for group A and group B | Compute means, standard error, and degrees of freedom. |
| Sum for each group | Compute group means. |
| Sum of squares for each group | Compute group variances and pooled variance. |

### Federated flow

```text
Input:
    y: numerical outcome
    x: grouping variable
    groupA, groupB: selected categories
    alpha: significance level
    alternative: two-sided, less, or greater

Step 1:
    At each site:
        select observations in groupA and groupB
        compute count, sum, and sum of squares for each group

Step 2:
    Aggregate group counts, sums, and sums of squares.

Step 3:
    Validate that both groups have observations and enough total degrees of freedom.

Step 4:
    Compute group means, pooled variance, standard error, t statistic,
    p-value, confidence interval, and Cohen's d.

Output:
    independent t-test summary
```

## Technical decisions

- The implementation uses pooled variance, equivalent to an equal-variance
  independent t-test.
- The selected `groupA` and `groupB` define the sign of the mean difference.
- One-sided alternatives replace one confidence bound with infinity.
- Cohen's d is computed from the pooled standard deviation.
- Missing-value handling is performed before the selected samples are tested.

## Outputs

| Field | Description |
|---|---|
| `t_stat` | t statistic. |
| `df` | Degrees of freedom. |
| `p` | P-value. |
| `mean_diff` | `mean(groupA) - mean(groupB)`. |
| `se_diff` | Standard error of the mean difference. |
| `ci_lower` | Lower confidence interval bound. |
| `ci_upper` | Upper confidence interval bound. |
| `cohens_d` | Standardized mean difference using pooled standard deviation. |

## Validation against state-of-the-art implementation

Standalone tests compare the result with `statsmodels.stats.weightstats`:

```text
CompareMeans(DescrStatsW(sample_a), DescrStatsW(sample_b))
    .ttest_ind(usevar="pooled", alternative="two-sided")
```

Reference behavior is aligned with an equal-variance independent t-test.

## Limitations and assumptions

- The outcome must be numerical.
- Exactly two selected groups are compared.
- The two groups are assumed independent.
- The pooled-variance test assumes comparable group variances.
- The method does not implement Welch's unequal-variance t-test.
