# Principal component analysis

Implementation: [decomposition/pca.py](../decomposition/pca.py).

Principal component analysis computes orthogonal directions that summarize
variance in selected numerical variables. The implementation standardizes
variables using aggregated means and sample standard deviations, then performs an
eigendecomposition of the aggregated covariance matrix.

## Library interface

`FederatedPCA(agg_client, copy=False)` supports `fit(X)`, `transform(X)`, and `fit_transform(X)`. `copy` is keyword-only. `fit` returns the fitted estimator, whose `explained_variance_` and `components_` contain eigenvalues and eigenvectors; `mean_`, `scale_`, and `n_samples_seen_` retain fitting statistics used when transforming new observations.

## Statistical method

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

## Numerical decisions

- PCA is covariance-based.
- Variables are standardized before eigendecomposition.
- Sample variance uses denominator `n - 1`.
- Negative variances from round-off are clipped to zero.
- Zero standard deviations are replaced by `1.0` during scaling to avoid
  division by zero.
- Eigenvectors are returned as rows ordered by descending eigenvalue.
- Component signs are anchored to the first loading tied for largest absolute magnitude, so repeated fitting is deterministic for well-separated eigenvalues.
- With `copy=False`, a writable floating-point input can be standardized in place; use `copy=True` to retain the original array.

## Reference comparison

Standalone tests compare against scikit-learn PCA pipelines:

```text
StandardScaler()
PCA(svd_solver="full")
```

Reference behavior is aligned with scikit-learn PCA after standardization. The
implementation uses sample variance (`ddof=1`) for scaling, while scikit-learn's
`StandardScaler` uses population variance (`ddof=0`), so tests account for this
difference.

## Limitations

- Inputs must be numerical.
- PCA is sensitive to scaling and outliers.
- `transform` returns local component scores; repeated or nearly equal eigenvalues can still allow rotations within their eigenspace.
