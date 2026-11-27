# Principal Component Analysis

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

## Method

The mathematical method, library API, aggregation flow, numerical decisions, reference comparisons, and limitations are documented in [Principal component analysis](../../exaflow/algorithms/federated/docs/pca.md).

## Application integration

Required missing-value preprocessing runs before fitting. The response contains components and eigenvalues rather than per-observation scores. The PCA component creator supports replay using the reusable preprocessing payload; that request/response contract belongs to the application.

## Outputs

| Field | Description |
|---|---|
| `title` | Result title. |
| `n_obs` | Number of observations used. |
| `eigenvalues` | Component variances in descending order. |
| `eigenvectors` | Principal component directions. |

## Compatibility

The separate PCA variant with per-variable transformations has been removed.
Standard `pca` remains available with its existing behavior and does not accept
a `data_transformation` parameter.
