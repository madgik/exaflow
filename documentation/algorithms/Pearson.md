# Pearson Correlation

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

Pearson correlation measures linear association between numerical variables. The
algorithm computes correlations, p-values, and confidence intervals for all
requested variable pairs.

## Inputs

### Required inputs

| Input | Description |
|---|---|
| `y` | Numerical variables for the primary correlation axis. |
| `x` | Optional numerical variables for the secondary axis. |

When `x` is omitted, correlations are computed among the variables in `y`.

### Parameters

| Parameter | Description | Default |
|---|---|---|
| `alpha` | Confidence level used for correlation intervals. | `0.95` |

## Statistical model

For variables `x` and `y`, Pearson's correlation is:

```text
r = cov(x, y) / (sd(x) sd(y))
```

The p-value is computed from the usual correlation t statistic with
`df = n - 2`. Confidence intervals use Fisher's z transform.

## Federated computation

The correlation matrix is computed without sharing row-level data. Each site
contributes sums, sums of squares, and cross-products for the requested variable
pairs.

### Aggregated quantities

| Quantity | Purpose |
|---|---|
| Number of observations | Compute degrees of freedom and confidence interval standard error. |
| `sum(x)` and `sum(y)` | Compute centered cross-products. |
| `sum(x^2)` and `sum(y^2)` | Compute variances. |
| `sum(x*y)` | Compute covariances for each variable pair. |

### Federated flow

```text
Input:
    Y: primary variables
    X: optional secondary variables
    alpha: confidence level

Step 1:
    Select X = Y when no secondary variables are provided.

Step 2:
    At each site, compute:
        n
        sums for X and Y
        sums of squares for X and Y
        cross-products X'Y

Step 3:
    Aggregate all sufficient statistics.

Step 4:
    Compute Pearson correlations for each pair.

Step 5:
    Compute p-values and Fisher-z confidence intervals.

Output:
    correlation, p-value, and confidence-interval matrices
```

## Technical decisions

- Correlations are clipped to `[-1, 1]` after computation.
- P-values use the beta-function form equivalent to the standard Pearson test.
- Perfect correlations receive p-value `0`.
- Zero-variance denominators produce correlation `0`.
- Missing values are handled before correlation by the required missing-values
  preprocessing step.

## Outputs

| Field | Description |
|---|---|
| `title` | Result title. |
| `n_obs` | Number of observations used. |
| `correlations` | Matrix of Pearson correlations. |
| `p_values` | Matrix of p-values. |
| `ci_lo` | Matrix of lower confidence interval bounds. |
| `ci_hi` | Matrix of upper confidence interval bounds. |

## Validation against state-of-the-art implementation

Standalone tests compare with `scipy.stats` Pearson correlation behavior.
Reference behavior is aligned with `scipy.stats.pearsonr` methodology for
correlation coefficients and p-values.

## Limitations and assumptions

- Variables must be numerical.
- Pearson correlation measures linear association only.
- Observations are assumed independent.
- Confidence intervals rely on Fisher-z asymptotic approximation.
- Pairwise missing-value behavior depends on the preprocessing selected before
  the algorithm runs.
