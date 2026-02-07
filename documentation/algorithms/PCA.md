## Principal Components Analysis

<b><h4>Aggregation Server</h4></b>
Some algorithms use the aggregation server to combine partial vectors (e.g., sums)
from workers into a single global result. The controller coordinates the flow and
workers send partial aggregates to the aggregation server via gRPC; the combined
result is then used in the algorithm’s global step.

#### Algorithm Description

The are usually two approaches for computing the *principal components*. The first is by
diagonalizing the covariance matrix, while the second is by *SVD* decomposition on the data
matrix *X*. In most implementations the second approach is preferred due to its numerical
stability. Here however, we took the first approach since it better fits with our privacy
requirements. Additionally, as a first step, data is centered and standardized.

![pseudo](images/pca_pseudocode.png)

#### Exareme3 Notes

- Data are centered and standardized before covariance aggregation.
- `pca_with_transformation` additionally applies preprocessing transformations
  before PCA (see implementation for details).

<b><h4>Algorithm Implementation</b></h4>

[PCA](../../exaflow/algorithms/exareme3/pca.py)

[PCA with Transformation](../../exaflow/algorithms/exareme3/pca_with_transformations.py)
