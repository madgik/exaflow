# One-sample t-test

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

The one-sample t-test compares the mean of a numerical variable with a reference
mean `mu`. It returns the t statistic, p-value, confidence interval, standard
error, and Cohen's d.

## Inputs

### Required inputs

| Input | Description |
|---|---|
| `y` | Numerical variable to test. |

### Parameters

| Parameter | Description | Default |
|---|---|---|
| `mu` | Mean value under the null hypothesis. | `0.0` |
| `alpha` | Significance level for confidence intervals. | `0.05` |
| `alt_hypothesis` | Alternative hypothesis: `two-sided`, `less`, or `greater`. | `two-sided` |

## Statistical model

The null hypothesis is:

```text
H0: mean(y) = mu
```

The test statistic is:

```text
t = (mean(y) - mu) / (s / sqrt(n))
```

with `df = n - 1`, where `s` is the sample standard deviation.

## Federated computation

The test is computed without sharing row-level data. Each site contributes
sample sufficient statistics, and the statistic is computed from aggregated
totals.

### Aggregated quantities

| Quantity | Purpose |
|---|---|
| Number of observations | Compute mean, standard error, and degrees of freedom. |
| Sum of values | Compute sample mean. |
| Sum of squared deviations from the global mean | Compute sample variance. |

### Federated flow

```text
Input:
    y: numerical variable
    mu: null mean
    alpha: significance level
    alternative: two-sided, less, or greater

Step 1:
    At each site:
        remove missing y values
        compute n and sum(y)

Step 2:
    Aggregate n and sum(y).
    Validate that the total number of observations is greater than one.
    Compute the global mean.

Step 3:
    At each site:
        compute sum((y - mean)^2)
    Aggregate it.

Step 4:
    Compute standard deviation, standard error, mean difference (mean - mu),
    t statistic, degrees of freedom, p-value, confidence interval, and
    Cohen's d.

Output:
    one-sample t-test summary
```

## Technical decisions

- Variance uses a second aggregation round of squared deviations from the
  global mean instead of raw moments, so it stays accurate when the mean is
  large relative to the spread.
- Confidence intervals use the Student t distribution and are reported for
  the mean difference `mean - mu`.
- One-sided alternatives replace one confidence bound with infinity.
- Cohen's d is computed as `(mean - mu) / standard_deviation`.
- Missing-value removal is handled before the sample reaches the test routine.

## Outputs

| Field | Description |
|---|---|
| `n_obs` | Number of observations used. |
| `std` | Sample standard deviation. |
| `t_stat` | t statistic. |
| `df` | Degrees of freedom. |
| `p` | P-value. |
| `mean_diff` | Sample mean minus `mu`. |
| `se_diff` | Standard error of the mean. |
| `ci_lower` | Lower confidence interval bound for `mean - mu`. |
| `ci_upper` | Upper confidence interval bound for `mean - mu`. |
| `cohens_d` | Standardized mean difference. |

## Validation against state-of-the-art implementation

Standalone tests compare the result with `statsmodels.stats.weightstats.DescrStatsW`
using:

```text
DescrStatsW(sample).ttest_mean(value=mu, alternative="two-sided")
```

The method is also aligned with standard one-sample t-test formulas used by
`scipy.stats`.

## Limitations and assumptions

- The variable must be numerical.
- At least two observations are required.
- Observations are assumed independent.
- The test assumes the sample mean is approximately t-distributed under the null.
- Cohen's d is undefined when the sample standard deviation is zero.
