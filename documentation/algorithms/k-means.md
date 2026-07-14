# K-means

## Table of contents

- [Overview](#overview)
- [Inputs](#inputs)
  - [Required inputs](#required-inputs)
  - [Parameters](#parameters)
- [Clustering method](#clustering-method)
- [Health use cases](#health-use-cases)
- [Federated computation](#federated-computation)
  - [Aggregated quantities](#aggregated-quantities)
  - [Federated flow](#federated-flow)
- [UI contract](#ui-contract)
- [Technical decisions](#technical-decisions)
- [Outputs](#outputs)
- [K-means as preprocessing](#k-means-as-preprocessing)
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
| `k_selection` | `manual` uses `k`; `elbow` evaluates `k_min..k_max` and selects a value from the inertia curve. | `manual` |
| `k` | Number of clusters when `k_selection=manual`. | `4` |
| `k_min` | Minimum number of clusters evaluated when `k_selection=elbow`. | `2` |
| `k_max` | Maximum number of clusters evaluated when `k_selection=elbow`. | `8` |
| `maxiter` | Maximum number of Lloyd iterations. | `100` |
| `tol` | Frobenius-norm convergence tolerance for center updates. | `0.0001` |
| `init_method` | `random_range` uses one random initialization from global feature ranges; `multi_start_random_range` evaluates multiple such initializations. | `random_range` |
| `n_init` | Number of initializations evaluated when `init_method=multi_start_random_range`. | `5` |

## Clustering method

The method follows Lloyd K-means:

```text
minimize sum_i ||x_i - c_{z_i}||^2
```

where `c_k` is a cluster center and `z_i` is the assigned cluster for
observation `i`.

The elbow method uses the same K-means objective, the within-cluster sum of
squared distances. It is not a user-defined clinical cost function. It helps
choose a plausible number of clusters, but the selected value still needs
clinical review.

Initialization uses aggregated global feature ranges, not raw patient rows.
With `random_range`, one set of initial centers is sampled uniformly from those
ranges. With `multi_start_random_range`, several independent random-range
initializations are fitted and the run with the lowest global inertia is kept.
This improves stability without selecting a real patient as an initial center.

## Health use cases

K-means is useful when the selected variables describe baseline patient
characteristics and the goal is exploratory grouping. Example variables:

| Variable | Meaning |
|---|---|
| `age` | Age at baseline. |
| `crp` | Inflammation marker. |
| `bmi` | Body mass index. |
| `systolic_bp` | Systolic blood pressure. |

Example interpretation:

- `cluster_0`: younger patients with lower `crp` and lower `systolic_bp`.
- `cluster_1`: older patients with higher `crp` and higher `systolic_bp`.

These clusters are not diagnoses. They are statistical summaries of selected
baseline variables. A clinician or domain expert must decide whether the
patterns are clinically meaningful.

K-means can also be used before another algorithm. For example, a request can
create `kmeans_cluster` from `age`, `crp`, and `bmi`, then use that derived
categorical variable as a covariate in linear regression, logistic regression,
or Cox regression. This supports questions such as whether a baseline cluster
is associated with an outcome after adjustment.

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
    privacy-safe cluster report
```

## UI contract

The UI should expose these controls:

| Control | Values | Notes |
|---|---|---|
| K selection | `manual`, `elbow` | Manual shows `k`; elbow shows `k_min` and `k_max`. |
| Number of clusters | integer `k` | Used only in manual mode. |
| Minimum K / Maximum K | integers `k_min`, `k_max` | Used only in elbow mode. |
| Maximum iterations | integer `maxiter` | Upper bound for Lloyd iterations. |
| Tolerance | real `tol` | Convergence threshold. |
| Initialization method | `random_range`, `multi_start_random_range` | Multi-start keeps the lowest-inertia random-range fit. |
| Number of initializations | integer `n_init` | Relevant only for `multi_start_random_range`. |

The result view should show:

- selected variables
- selected `k`
- size interval per cluster, not exact count
- cluster center only when privacy allows it
- textual profile and interpretation per visible cluster
- warnings for suppressed clusters, non-convergence, and empty clusters

## Technical decisions

- Initialization samples uniformly from aggregated feature ranges.
- `multi_start_random_range` repeats range-based initialization with different
  seeds and keeps the fitted model with the lowest global inertia.
- Random seed `123` is used for reproducibility.
- Squared Euclidean distance is used for assignment.
- Empty clusters are reset to the origin.
- Convergence uses the Frobenius norm of the center update.
- Cluster sizes are reported as intervals rather than exact counts.
- Centers and textual profiles are hidden for clusters below the privacy
  minimum-row threshold.

## Outputs

| Field | Description |
|---|---|
| `title` | Result title. |
| `result_type` | `privacy_safe_cluster_report`. |
| `variables` | Variables used for clustering. |
| `k_selection` | `manual` or `elbow`. |
| `selected_k` | Number of fitted clusters. |
| `initialization_method` | Initialization method used by the fitted model. |
| `n_init` | Number of initializations actually evaluated. |
| `selected_initialization` | Zero-based index of the initialization kept. |
| `n_obs_interval` | Total observation count interval. |
| `center_definition` | Explains that centers are mean profiles, not patients. |
| `intended_use` | Supported high-level uses. |
| `privacy_note` | Summary of privacy masking. |
| `clusters` | Per-cluster interval, center/profile when allowed, and interpretation. |
| `elbow` | Elbow diagnostics when `k_selection=elbow`. |
| `warnings` | Privacy, convergence, or empty-cluster warnings. |
| `limitations` | Clinical and statistical limitations. |

The result does not include per-observation cluster labels.

## K-means as preprocessing

The `kmeans_cluster_creator` preprocessing step fits K-means and creates a new
categorical variable for downstream algorithms.

Output modes:

| Mode | Output | Use case |
|---|---|---|
| `full` | `cluster_0`, `cluster_1`, ..., `cluster_k-1` | Use the complete cluster assignment as one categorical covariate. |
| `binary` | `yes` / `no` for one selected cluster | Test membership in one clinically reviewed cluster. |
| `subset` | selected clusters plus `other` | Keep a small set of clusters explicit and combine the rest. |

If `subset` contains exactly one selected cluster, the step automatically
creates a binary `yes` / `no` variable.

The preprocessing step validates privacy at runtime. It rejects outputs where
an exposed category would be smaller than the privacy threshold.

## Validation against state-of-the-art implementation

The method is aligned with classical Lloyd K-means as exposed by:

```text
sklearn.cluster.KMeans(algorithm="lloyd")
```

Important differences from common scikit-learn defaults:

| Aspect | This method | scikit-learn default |
|---|---|---|
| Initialization | Uniform sampling from aggregated feature ranges | `k-means++` |
| Number of initializations | One, or multiple with `multi_start_random_range` | Depends on `n_init` |
| Empty clusters | Reset center to zero | Internal reassignment behavior |
| Objective | Lloyd K-means | Lloyd K-means by default in current releases |

## Limitations and assumptions

- Only numerical variables are supported.
- Feature scaling strongly affects results.
- The solution can be a local optimum.
- Empty-cluster handling can affect final centers.
- The number of clusters is selected manually or by elbow; neither choice is
  clinical validation.
- Outliers can strongly influence centers.
- K-means does not infer diagnosis, prognosis, treatment effect, or causality.
- Clusters produced from baseline covariates should be treated as hypotheses or
  covariates, not as final clinical labels.
