# K-means clustering

Implementation: [cluster/kmeans.py](../cluster/kmeans.py).

K-means partitions numeric observations using squared Euclidean distance and aggregated cluster sufficient statistics.

## Library interface

`FederatedKMeans(n_clusters=..., init_method="random_range", n_init=1, tol=1e-4, maxiter=100, random_state=123).fit(x, y=None, *, agg_client, feature_names=None)` fits numerical features. Constructor settings are keyword-only. `select_k(x, *, k_min, k_max, agg_client, ...)` evaluates a candidate range for elbow selection.

`FederatedKMeansResults` includes centers, exact counts, local labels, inertia, iterations, and convergence/selection diagnostics. These are library results; callers decide which fitted statistics to release. The library default is one initialization; multi-start behavior is explicit.

## Statistical method

The method follows Lloyd K-means:

```text
minimize sum_i ||x_i - c_{z_i}||^2
```

where `c_k` is a cluster center and `z_i` is the assigned cluster for
observation `i`.

Range-based initialization samples from global feature minima and maxima. Multi-start fitting keeps the run with the lowest global inertia. Elbow selection evaluates this same objective over a candidate cluster range.

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
Receive numeric X and a shared cluster/initialization configuration.
Aggregate counts and feature ranges; sample initial centers from these ranges.
For each requested initialization:
    assign local rows to nearest centers by squared Euclidean distance
    aggregate cluster counts and feature sums
    update centers as sums / counts; reset empty clusters to zero
    repeat until center-update norm <= tol or maxiter is reached
    compute global inertia
Keep the initialization with the lowest inertia.
Return fitted centers, exact counts, local assignments, and diagnostics.
For select_k, repeat fitting over the candidate range and locate the elbow.
```

## Numerical decisions

- Initialization samples uniformly from aggregated feature ranges.
- `multi_start_random_range` repeats range-based initialization with different
  seeds and keeps the fitted model with the lowest global inertia.
- Random seed `123` is used for reproducibility.
- Squared Euclidean distance is used for assignment.
- Empty clusters are reset to the origin.
- Convergence uses the Frobenius norm of the center update.

## Reference comparison

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

## Limitations

- Only numerical variables are supported.
- Feature scaling strongly affects results.
- The solution can be a local optimum.
- Empty-cluster handling can affect final centers.
- Outliers can strongly influence centers.
