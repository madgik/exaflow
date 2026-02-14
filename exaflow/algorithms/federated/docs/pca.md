## Principal Component Analysis (FederatedPCA)

### Name

**Principal Component Analysis (FederatedPCA)**

### Type

**Dimensionality Reduction** (unsupervised)

### Goal (Why we need it)

PCA transforms high-dimensional data into a lower-dimensional representation by finding **principal components** (directions of maximum variance).
In a federated setting, we want the *same* principal components and explained variance as centralized PCA **without sharing raw data**, using only aggregated moments (mean, variance, covariance via Gram matrix).

### When to use

Use PCA when:

* you need **dimensionality reduction** to reduce feature space
* you want to remove multicollinearity
* you want to visualize high-dimensional data (e.g., first 2-3 components)
* features are numerical and ideally normalized
* you can accept linear combinations of features

### When NOT to use

Avoid / be careful when:

* features are categorical without proper encoding
* features have very different scales and are not standardized (PCA is scale-sensitive)
* you need interpretable features (principal components are linear combinations)
* non-linear relationships dominate (consider kernel PCA or other methods)
* sparse data where standardization amplifies noise
* privacy is critical beyond aggregate statistics (components can leak distribution info)

---

### Inputs / Outputs

| Item           | Description                                                      |
| -------------- | ---------------------------------------------------------------- |
| **X**          | Local data matrix, shape `(n_local, p)`                          |
| **copy**       | Whether to copy data before centering/scaling (default: `False`) |
| **agg_client** | Federated aggregation client                                     |

**Outputs**

* `components_`: principal components (eigenvectors), shape `(p, p)`
* `explained_variance_`: variance explained by each component
* `mean_`: feature means used for centering
* `scale_`: feature standard deviations used for scaling
* `n_samples_seen_`: total number of observations across all clients
* `transform(X)`: project data onto principal components
* `fit_transform(X)`: fit and transform in one step

### Key Differences from scikit-learn

| Aspect                 | scikit-learn          | MIP Federated Implementation  |
| ---------------------- | --------------------- | ----------------------------- |
| Data access            | Full centralized data | Data remains local per client |
| Algorithm              | SVD or eigendecomp    | Eigendecomposition of cov     |
| Covariance computation | Direct from X         | Via aggregated Gram matrix    |
| Centering              | Local computation     | Federated mean                |
| Scaling                | Local computation     | Federated std dev             |
| Performance            | CPU/memory bound      | Network + aggregation bound   |
| n_components parameter | Supports selection    | Returns all components        |

### Approximation vs Exactness

| Component                  | scikit-learn | MIP              |
| -------------------------- | ------------ | ---------------- |
| Mean (centering)           | Exact        | Exact            |
| Variance (scaling)         | Exact        | Exact            |
| Covariance matrix          | Exact        | Exact            |
| Eigenvalues                | Exact        | Exact            |
| Eigenvectors (components)  | Exact        | Exact            |
| Explained variance         | Exact        | Exact            |
| Transformation             | Exact        | Exact            |

