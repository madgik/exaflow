# K-means

Runtime wrapper: [cluster/kmeans.py](../../exaflow/algorithms/exareme3/cluster/kmeans.py).

K-means owns its replay models, parameter builder, and fingerprint helpers in
`cluster/kmeans.py`. The adjacent cluster column creator imports the fitted
result contract from that module.

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
| `enumeration_prefix` | Prefix for cluster identifiers, numbered from 1 through the fitted count. | `cluster_` |
| `k_min` | Minimum number of clusters evaluated when `k_selection=elbow`. | `2` |
| `k_max` | Maximum number of clusters evaluated when `k_selection=elbow`. | `8` |
| `maxiter` | Maximum number of Lloyd iterations. | `100` |
| `tol` | Frobenius-norm convergence tolerance for center updates. | `0.0001` |
| `init_method` | `random_range` uses one random initialization from global feature ranges; `multi_start_random_range` evaluates multiple such initializations. | `random_range` |
| `n_init` | Number of initializations evaluated when `init_method=multi_start_random_range`. | `5` |
| `random_state` | Seed used for reproducible center initialization. | `123` |

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

- `cluster_1`: younger patients with lower `crp` and lower `systolic_bp`.
- `cluster_2`: older patients with higher `crp` and higher `systolic_bp`.

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
| Cluster category prefix | text `enumeration_prefix` | Defaults to `cluster_`. |
| Minimum K / Maximum K | integers `k_min`, `k_max` | Used only in elbow mode. |
| Maximum iterations | integer `maxiter` | Upper bound for Lloyd iterations. |
| Tolerance | real `tol` | Convergence threshold. |
| Initialization method | `random_range`, `multi_start_random_range` | Multi-start keeps the lowest-inertia random-range fit. |
| Number of initializations | integer `n_init` | Relevant only for `multi_start_random_range`. |

The result view should show:

- selected variables
- selected `k`
- size interval per cluster, not exact count
- cluster center for every fitted cluster
- textual profile and interpretation per cluster
- warnings for non-convergence and empty clusters

## Technical decisions

- Initialization samples uniformly from aggregated feature ranges.
- `multi_start_random_range` repeats range-based initialization with different
  seeds and keeps the fitted model with the lowest global inertia.
- Random seed `123` is used for reproducibility.
- Squared Euclidean distance is used for assignment.
- Empty clusters are reset to the origin.
- Convergence uses the Frobenius norm of the center update.
- Cluster sizes are reported as intervals rather than exact counts.
- Centers and textual profiles are returned for every fitted cluster; only the
  cluster-size interval is privacy-masked.

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
| `clusters` | Per-cluster privacy-safe interval, center, profile, and interpretation. |
| `elbow` | Elbow diagnostics when `k_selection=elbow`. |
| `reusable_preprocessing` | Replay contract for creating a selected K-means categorical variable downstream. |
| `warnings` | Privacy, convergence, or empty-cluster warnings. |
| `limitations` | Clinical and statistical limitations. |

The result does not include per-observation cluster labels.

## K-means as preprocessing

The `kmeans_cluster_creator` preprocessing step replays a fitted K-means model
and creates a new categorical variable for downstream algorithms.

The only supported output mode is `full`, creating one categorical column with
one category per fitted cluster. With the default prefix, identifiers are
`cluster_1`, `cluster_2`, ..., `cluster_k`. A custom `enumeration_prefix` changes
these identifiers consistently in the report, stored centers, and created column.
Internal cluster indices remain zero-based.

The reusable preprocessing contract retains the
cluster variables, exact fitted centers, source data context, available outputs,
and cluster choices, and includes `enumeration_prefix`. It contains neither
row-level assignments nor random seeds.

Use `build_kmeans_preprocessing_parameters(reusable, new_column_names=["group"])`
to populate the creator request. This supplies the complete
`reusable_preprocessing` payload plus `k` (the actual fitted center count) and
`enumeration_prefix`. Users choose only the new column name; the count and prefix
are copied from the fitted model. In elbow mode, the actual count corresponds to
`selected_k`, not the original manual `k` setting.

The creator's output enumeration specification is:

```json
{
  "source_parameter_name": "k",
  "source_type": "prefixed_range",
  "prefix_parameter_name": "enumeration_prefix"
}
```

Clients generate categories by concatenating the prefix with numbers 1 through
`k`, without inspecting nested centers. The creator validates that these
identifiers exactly match its replay model. It also requires the original data
model, datasets, filters, and clustering features. Assignment uses generated
numeric order, including for more than nine clusters or prefixes containing
underscores.

Compatibility: this intentionally replaces zero-based identifiers and scalar
`code` creator requests. Replay payloads must satisfy the current model directly;
there is no schema-version field, migration, or legacy payload support. Update
clients and service together; rollback requires restoring both the old service
and its matching consumers/model payloads.

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
