# Principal Component Analysis

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
- [Related page](#related-page)

## Overview

Principal component analysis computes orthogonal directions that summarize
variance in selected numerical variables. The implementation standardizes
variables using aggregated means and sample standard deviations, then performs an
eigendecomposition of the aggregated covariance matrix.

## Inputs

### Required inputs

| Input | Description |
|---|---|
| `y` | Numerical variables used to compute components. |

### Parameters

No user parameters are exposed for this base PCA page.

## Decomposition method

After centering and scaling each feature, PCA solves:

```text
cov(X_standardized) v_j = lambda_j v_j
```

Eigenvectors `v_j` are component directions, and eigenvalues `lambda_j`
describe variance explained by each component.

## Federated computation

The decomposition is computed without sharing row-level data. Sites contribute
moments needed for global standardization and the standardized Gram matrix.

### Aggregated quantities

| Quantity | Purpose |
|---|---|
| Number of observations | Compute means, sample variances, and covariance denominator. |
| Feature sums | Compute global means. |
| Feature sums of squares | Compute global sample variances. |
| Standardized Gram matrix | Compute the covariance matrix for eigendecomposition. |

### Federated flow

```text
Input:
    X: numerical feature matrix

Step 1:
    At each site, compute:
        n
        sum of each feature
        sum of squares for each feature

Step 2:
    Aggregate n, sums, and sums of squares.

Step 3:
    Compute global means and sample standard deviations.

Step 4:
    At each site:
        center and scale X using global statistics
        compute the local standardized Gram matrix

Step 5:
    Aggregate the Gram matrix.

Step 6:
    Compute covariance = Gram / (n - 1).

Step 7:
    Eigendecompose covariance and sort eigenvalues descending.

Output:
    observation count, eigenvalues, and eigenvectors
```

## Technical decisions

- PCA is covariance-based.
- Variables are standardized before eigendecomposition.
- Sample variance uses denominator `n - 1`.
- Negative variances from round-off are clipped to zero.
- Zero standard deviations are replaced by `1.0` during scaling to avoid
  division by zero.
- Eigenvectors are returned as rows ordered by descending eigenvalue.

## Outputs

| Field | Description |
|---|---|
| `title` | Result title. |
| `n_obs` | Number of observations used. |
| `eigenvalues` | Component variances in descending order. |
| `eigenvectors` | Principal component directions. |

## Validation against state-of-the-art implementation

Standalone tests compare against scikit-learn PCA pipelines:

```text
StandardScaler()
PCA(svd_solver="full")
```

Reference behavior is aligned with scikit-learn PCA after standardization. The
implementation uses sample variance (`ddof=1`) for scaling, while scikit-learn's
`StandardScaler` uses population variance (`ddof=0`), so tests account for this
difference.

## Limitations and assumptions

- Inputs must be numerical.
- PCA is sensitive to scaling and outliers.
- Missing values are handled before fitting.
- The result returns components, not transformed scores for each observation.

## Related page

See [PCA with transformations](PCAWithTransformation.md) for the variant that
applies log, exponential, centering, or standardization transformations before
PCA.
