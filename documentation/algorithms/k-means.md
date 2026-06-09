# K-means

## Table of contents

- [Overview](#overview)
- [Inputs](#inputs)
  - [Required inputs](#required-inputs)
  - [Parameters](#parameters)
- [Clustering method](#clustering-method)
- [Federated computation](#federated-computation)
  - [Aggregated quantities](#aggregated-quantities)
  - [Federated flow](#federated-flow)
- [Technical decisions](#technical-decisions)
- [Outputs](#outputs)
- [Validation against state-of-the-art implementation](#validation-against-state-of-the-art-implementation)
- [Limitations and assumptions](#limitations-and-assumptions)

## Overview

K-means partitions observations into `k` clusters using numerical variables.
Each cluster is represented by a center, and observations are assigned to the
nearest center by squared Euclidean distance.

## Inputs

### Required inputs

| Input | Description |
|---|---|
| `y` | Numerical variables used for clustering. |

### Parameters

| Parameter | Description | Default |
|---|---|---|
| `k` | Number of clusters. | `4` |
| `maxiter` | Maximum number of Lloyd iterations. | `1` |
| `tol` | Frobenius-norm convergence tolerance for center updates. | `0.01` |

## Clustering method

The method follows Lloyd K-means:

```text
minimize sum_i ||x_i - c_{z_i}||^2
```

where `c_k` is a cluster center and `z_i` is the assigned cluster for
observation `i`.

## Federated computation

The algorithm is computed without sharing row-level data. Each site assigns its
observations to the current centers and contributes cluster-wise sums and
counts.

### Aggregated quantities

| Quantity | Purpose |
|---|---|
| Number of observations | Report sample size and handle empty input. |
| Feature minima and maxima | Initialize centers from global feature ranges. |
| Cluster-wise feature sums | Update cluster centers. |
| Cluster-wise counts | Convert sums to means. |

### Federated flow

```text
Input:
    X: numerical data matrix
    k: number of clusters
    maxiter: maximum iterations
    tol: convergence tolerance

Step 1:
    Aggregate the total number of observations.

Step 2:
    Aggregate feature-wise minima and maxima.

Step 3:
    Initialize k centers uniformly between global minima and maxima.

Step 4:
    Repeat up to maxiter:
        each site assigns observations to nearest centers
        each site computes cluster-wise feature sums and counts
        aggregate sums and counts
        update each center as sum / count
        reset empty-cluster centers to zero
        stop if the center-update norm is <= tol

Output:
    total observation count and fitted centers
```

## Technical decisions

- Initialization samples uniformly from aggregated feature ranges.
- Random seed `123` is used for reproducibility.
- Squared Euclidean distance is used for assignment.
- Empty clusters are reset to the origin.
- Convergence uses the Frobenius norm of the center update.
- The default `maxiter` is intentionally small and may need to be increased for
  practical clustering.

## Outputs

| Field | Description |
|---|---|
| `title` | Result title. |
| `n_obs` | Number of observations used for fitting. |
| `centers` | Fitted cluster centers. |

The result does not include per-observation cluster labels.

## Validation against state-of-the-art implementation

The method is aligned with classical Lloyd K-means as exposed by:

```text
sklearn.cluster.KMeans(algorithm="lloyd")
```

Important differences from common scikit-learn defaults:

| Aspect | This method | scikit-learn default |
|---|---|---|
| Initialization | Uniform sampling from aggregated feature ranges | `k-means++` |
| Number of initializations | One | Depends on `n_init` |
| Empty clusters | Reset center to zero | Internal reassignment behavior |
| Objective | Lloyd K-means | Lloyd K-means by default in current releases |

## Limitations and assumptions

- Only numerical variables are supported.
- Feature scaling strongly affects results.
- The solution can be a local optimum.
- Empty-cluster handling can affect final centers.
- The number of clusters must be selected before fitting.
- Outliers can strongly influence centers.
