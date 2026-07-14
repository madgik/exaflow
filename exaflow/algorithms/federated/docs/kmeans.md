## KMeans (FederatedKMeans)

### Name

**KMeans (FederatedKMeans)**

### Type

**Statistical Model** (unsupervised clustering)

### Goal (Why we need it)

KMeans groups observations into **K clusters** by learning **K cluster centers (centroids)**.
In a federated setting, we want the *same* clustering result as centralized KMeans **without sharing raw data**, using only aggregated statistics (sums, counts, min/max).

### When to use

Use KMeans when:

- you need **unsupervised clustering** (no labels)
- features are **numerical** (continuous or numeric-encoded)
- you can accept iterative training (multiple rounds)
- you want to discover natural groupings in distributed data

### When NOT to use

Avoid / be careful when:

- variables are categorical without proper encoding
- features have very different scales and are not scaled (KMeans is scale-sensitive)
- heavy outliers dominate (consider clipping / robust scaling)
- you need strong privacy beyond aggregates (centroids can leak some distribution info)
- K is unknown (KMeans requires specifying number of clusters)
- clusters have non-spherical shapes (KMeans assumes spherical clusters)

______________________________________________________________________

### Inputs / Outputs

| Item | Description |
| ------------------ | ----------------------------------------------------------- |
| **X** | Local data matrix, shape `(n_local, p)` |
| **n_clusters (K)** | Number of clusters |
| **tol** | Convergence threshold on centroid movement (Frobenius norm) |
| **maxiter** | Maximum Lloyd iterations |
| **random_state** | Seed for centroid initialization |
| **init_method** | `random_range` or `multi_start_random_range` |
| **n_init** | Number of initializations for multi-start fitting |

**Outputs / fitted state**

Public wrappers should expose only privacy-safe summaries. The federated core
keeps the following fitted state for internal reporter and preprocessing use:

- `cluster_centers_`: list of `K` statistical mean centers, each length `p`
- `cluster_counts_`: exact global cluster counts, internal only
- `labels_`: local worker labels, internal only
- `n_obs_`: exact global number of observations, internal only
- `inertia_`: global within-cluster sum of squared distances
- `cluster_inertia_`: per-cluster inertia, internal only
- `n_iter_`: number of Lloyd iterations used
- `converged_`: convergence flag
- `empty_clusters_`: cluster indexes with zero assigned observations
- `init_method_`: initialization strategy used
- `n_init_`: number of initializations actually evaluated
- `best_init_`: zero-based initialization index with the lowest inertia

### Key Differences from scikit-learn

| Aspect | scikit-learn | MIP Federated Implementation |
| ----------------- | --------------------- | ----------------------------- |
| Data access | Full centralized data | Data remains local per client |
| Initialization | K-means++ | Uniform random in min/max box |
| Multi-start | Supported through `n_init` | Supported through `multi_start_random_range` |
| Empty clusters | Multiple strategies | Reset to origin |
| Aggregation | In-memory operations | Federated primitives only |
| Performance | CPU/memory bound | Network + aggregation bound |
| Public result | Raw model attributes | Privacy-safe wrapper report |

### Approximation vs Exactness

| Component | scikit-learn | MIP |
| ---------------- | ------------ | ------------------- |
| Lloyd iterations | Exact | Exact |
| Mean (centroid) | Exact | Exact |
| Sum / count | Exact | Exact |
| Min / max | Exact | Exact |
| Initialization | K-means++ | Random uniform |
| Multi-start | Optional | Optional with `multi_start_random_range` |
| Empty clusters | Various | Reset to zero |
| Convergence | Exact | Exact (Frob. norm) |

### Privacy boundary

The federated core computes exact counts, centers, labels, and inertia because
they are needed for fitting, reporting, and preprocessing. These values are not
all public API output. The Exareme3 K-means wrapper masks cluster sizes as
intervals and suppresses centers/profiles for clusters below the privacy
threshold. The `kmeans_cluster_creator` preprocessing step validates that any
exposed categorical output class is large enough before creating the derived
column.

### Multi-start random range

`multi_start_random_range` is the privacy-safe initialization improvement used
before any K-means++ implementation. It runs Lloyd K-means multiple times with
different random seeds sampled from the global feature ranges, computes global
inertia for each run, and keeps the fitted model with the lowest inertia. It
does not select raw patient rows as centers.
