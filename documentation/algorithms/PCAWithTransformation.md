# PCA with Transformations

## Table of contents

- [Overview](#overview)
- [Inputs](#inputs)
  - [Required inputs](#required-inputs)
  - [Parameters](#parameters)
- [Decomposition method](#decomposition-method)
- [Federated computation](#federated-computation)
  - [Aggregated quantities](#aggregated-quantities)
  - [Federated flow](#federated-flow)
- [Technical decisions](#technical-decisions)
- [Outputs](#outputs)
- [Validation against state-of-the-art implementation](#validation-against-state-of-the-art-implementation)
- [Limitations and assumptions](#limitations-and-assumptions)

## Overview

PCA with transformations applies selected per-variable transformations before
computing principal components. Supported transformations are `log`, `exp`,
`center`, and `standardize`.

## Inputs

### Required inputs

| Input | Description |
|---|---|
| `y` | Numerical variables used to compute components. |

### Parameters

| Parameter | Description | Default |
|---|---|---|
| `data_transformation` | Dictionary mapping transformation names to selected variables. | `{}` |

Supported transformation keys:

| Key | Behavior |
|---|---|
| `log` | Apply natural logarithm locally; values must be positive. |
| `exp` | Apply exponential locally. |
| `center` | Subtract aggregated mean. |
| `standardize` | Subtract aggregated mean and divide by aggregated sample standard deviation. |

## Decomposition method

After requested transformations are applied, the algorithm delegates to the base
PCA method: global standardization followed by eigendecomposition of the
aggregated covariance matrix.

## Federated computation

The computation is performed without sharing row-level data. Log and exponential
transforms are applied independently at each site. Centering and standardization
use aggregated means and sample standard deviations. Each standardization uses
two aggregation rounds: the global mean comes first, then sums of squared
deviations from that mean. Sample variances are never derived from raw moments
(`sum(x^2) - n * mean^2`), which lose all precision when the mean is large
relative to the spread.

### Aggregated quantities

| Quantity | Purpose |
|---|---|
| Number of observations | Compute means, variances, and covariance denominator. |
| Feature sums after log/exp transforms | Compute global means for center/standardize. |
| Feature sums of squared deviations from the global mean after log/exp transforms | Compute global sample standard deviations. |
| Standardized Gram matrix | Compute PCA covariance matrix. |

### Federated flow

```text
Input:
    X: numerical feature matrix
    data_transformation: requested transformations

Step 1:
    Validate transformation names.

Step 2:
    At each site:
        apply log transforms to selected positive variables
        apply exp transforms to selected variables

Step 3:
    If center or standardize is requested:
        aggregate n and sums after log/exp transforms
        compute global means = sums / n
        aggregate sum((x - global_mean)^2) for each variable
        compute sample standard deviations = sqrt(sum_sq_dev / (n - 1))
        center or standardize selected variables

Step 4:
    Run base PCA:
        aggregate n and sums, then compute global means
        aggregate squared deviations, then compute standard deviations
        aggregate the standardized Gram matrix
        eigendecompose covariance

Output:
    observation count, eigenvalues, and eigenvectors
```

## Technical decisions

- Unknown transformation names raise an error.
- Log transformation rejects non-positive values.
- Standardization rejects explicitly selected zero-variance columns.
- Centering and standardization are computed after log/exp transformations.
- The base PCA routine standardizes variables again before eigendecomposition.

## Outputs

| Field | Description |
|---|---|
| `title` | Result title. |
| `n_obs` | Number of observations used. |
| `eigenvalues` | Component variances in descending order. |
| `eigenvectors` | Principal component directions. |

## Validation against state-of-the-art implementation

Validation follows the base PCA comparison with scikit-learn PCA methodology,
after applying the requested transformations. Reference behavior is aligned with
covariance-based PCA, with the same sample-variance scaling caveat documented in
[PCA](PCA.md).

## Limitations and assumptions

- Inputs must be numerical.
- Log-transformed values must be strictly positive before transformation.
- Standardizing a zero-variance selected variable is invalid.
- The result returns components, not transformed scores for each observation.
