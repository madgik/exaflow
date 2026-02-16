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

* you need **unsupervised clustering** (no labels)
* features are **numerical** (continuous or numeric-encoded)
* you can accept iterative training (multiple rounds)
* you want to discover natural groupings in distributed data

### When NOT to use

Avoid / be careful when:

* variables are categorical without proper encoding
* features have very different scales and are not scaled (KMeans is scale-sensitive)
* heavy outliers dominate (consider clipping / robust scaling)
* you need strong privacy beyond aggregates (centroids can leak some distribution info)
* K is unknown (KMeans requires specifying number of clusters)
* clusters have non-spherical shapes (KMeans assumes spherical clusters)

---

### Inputs / Outputs

| Item               | Description                                                 |
| ------------------ | ----------------------------------------------------------- |
| **X**              | Local data matrix, shape `(n_local, p)`                     |
| **n_clusters (K)** | Number of clusters                                          |
| **tol**            | Convergence threshold on centroid movement (Frobenius norm) |
| **maxiter**        | Maximum Lloyd iterations                                    |
| **random_state**   | Seed for centroid initialization                            |

**Outputs**

* `cluster_centers_`: list of `K` centers, each length `p`
* `n_obs_`: global number of observations (sum across clients)

### Key Differences from scikit-learn

| Aspect            | scikit-learn          | MIP Federated Implementation  |
| ----------------- | --------------------- | ----------------------------- |
| Data access       | Full centralized data | Data remains local per client |
| Initialization    | K-means++             | Uniform random in min/max box |
| Empty clusters    | Multiple strategies   | Reset to origin               |
| Aggregation       | In-memory operations  | Federated primitives only     |
| Performance       | CPU/memory bound      | Network + aggregation bound   |

### Approximation vs Exactness

| Component        | scikit-learn | MIP                 |
| ---------------- | ------------ | ------------------- |
| Lloyd iterations | Exact        | Exact               |
| Mean (centroid)  | Exact        | Exact               |
| Sum / count      | Exact        | Exact               |
| Min / max        | Exact        | Exact               |
| Initialization   | K-means++    | Random uniform      |
| Empty clusters   | Various      | Reset to zero       |
| Convergence      | Exact        | Exact (Frob. norm)  |

