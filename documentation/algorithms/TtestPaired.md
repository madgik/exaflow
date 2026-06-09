# Paired t-test

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

The paired t-test compares two related numerical measurements by testing whether
the mean paired difference is zero. It returns the t statistic, p-value,
confidence interval, standard error, and Cohen's d for the paired differences.

## Inputs

### Required inputs

| Input | Description |
|---|---|
| `y` | First numerical measurement. |
| `x` | Second numerical measurement. |

### Parameters

| Parameter | Description | Default |
|---|---|---|
| `alpha` | Significance level for confidence intervals. | `0.05` |
| `alt_hypothesis` | Alternative hypothesis: `two-sided`, `less`, or `greater`. | `two-sided` |

## Statistical model

For paired measurements `(a_i, b_i)`, define:

```text
d_i = a_i - b_i
```

The null hypothesis is:

```text
H0: mean(d) = 0
```

The test statistic is:

```text
t = mean(d) / (s_d / sqrt(n))
df = n - 1
```

## Federated computation

The test is computed without sharing row-level data. Each site contributes
sufficient statistics for the paired differences.

### Aggregated quantities

| Quantity | Purpose |
|---|---|
| Number of pairs | Compute degrees of freedom and standard error. |
| Sum of first and second measurements | Compute measurement means. |
| Sum of paired differences | Compute mean difference. |
| Sum of squared paired differences | Compute standard deviation of differences. |
| Sum of squared first and second measurements | Compute supporting summary statistics. |

### Federated flow

```text
Input:
    y: first measurement
    x: second measurement
    alpha: significance level
    alternative: two-sided, less, or greater

Step 1:
    At each site:
        keep complete pairs
        compute paired differences
        compute count, sums, and squared sums

Step 2:
    Aggregate paired-difference sufficient statistics.

Step 3:
    Validate that more than one pair is available.

Step 4:
    Compute mean difference, standard deviation of differences,
    standard error, t statistic, p-value, confidence interval, and Cohen's d.

Output:
    paired t-test summary
```

## Technical decisions

- The sign of the reported mean difference is `y - x`.
- One-sided alternatives replace one confidence bound with infinity.
- Cohen's d is computed as mean paired difference divided by the standard
  deviation of paired differences.
- The paired samples must have equal length after preprocessing.

## Outputs

| Field | Description |
|---|---|
| `t_stat` | t statistic. |
| `df` | Degrees of freedom. |
| `p` | P-value. |
| `mean_diff` | Mean paired difference. |
| `se_diff` | Standard error of the mean difference. |
| `ci_lower` | Lower confidence interval bound. |
| `ci_upper` | Upper confidence interval bound. |
| `cohens_d` | Standardized paired mean difference. |

## Validation against state-of-the-art implementation

Standalone tests compare the result with:

```text
scipy.stats.ttest_rel(sample_x, sample_y, alternative="two-sided")
```

Reference behavior is aligned with SciPy's paired t-test for the t statistic and
p-value, with additional confidence interval and effect-size reporting.

## Limitations and assumptions

- Both measurements must be numerical.
- Measurements must be paired one-to-one.
- At least two complete pairs are required.
- Paired differences are assumed approximately normally distributed.
- Observed pairs are assumed independent of one another.
