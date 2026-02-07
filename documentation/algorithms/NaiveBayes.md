<b><h2><center>Naive Bayes Classifier</center></h1></b>

<b><h4>Aggregation Server</h4></b>
Some algorithms use the aggregation server to combine partial vectors (e.g., sums)
from workers into a single global result. The controller coordinates the flow and
workers send partial aggregates to the aggregation server via gRPC; the combined
result is then used in the algorithm’s global step.

<b><h4> Notation </h4></b>
Each local dataset *D<sup>(l)</sup>*, where *l*=1,...,*L*, is represented as a matrix of size *n* x *p*, where *L* is the number of medical centers, *n* is the number of points (patients) and *p* is the number of attributes. The elements of the above matrix can either be continuous or discrete (categorical).

In each local dataset, the independent attributes are denoted as a matrix *X<sup>(l)</sup>* and the dependent variable is denoted as a vector *y<sup>(l)</sup>*. *x*<sub>(*ij*)</sub><sup>(*l*)</sup> is the value of the *i*<sup>(*th*)</sup> patient of the *j*<sup>(*th*)</sup> attribute in the *l*<sup>(*th*)</sup> hospital, while *x*<sub>(*j*)</sub><sup>(*l*)</sup> denotes the vector of the *j*<sup>(*th*)</sup> attribute in the *l*<sup>(*th*)</sup> hospital. For categorical attributes, we use the notation *C*<sub>m</sub> <img src="https://render.githubusercontent.com/render/math?math=\epsilon"> { *C*<sub>1</sub>, *C*<sub>2</sub>, ..., *C*<sub>M</sub>} for their domain.

<b><h4> Algorithm Description </h4></b>
In Naive Bayes algorithm the attributes of *X* can be both categorical and continuous, while the *y* is always categorical. Once we have the likelihood terms from the training procedure we can compute the maximum a posteriori probability for the class of a new query datapoint *q* with the following procedure:

![pseudo](images/nb_train_pseudocode.png)

![pseudo](images/nb_predict_pseudocode.png)

<b><h4>Exareme3 Notes</h4></b>

- Two variants are implemented:
  - **Categorical NB**: all predictors are treated as categorical and encoded
    via a federated ordinal encoder using the metadata enumerations.
  - **Gaussian NB**: predictors are numeric and modeled with class-conditional
    Gaussians (no preprocessing).
- CV versions are available for both variants and report per-fold confusion
  matrices and aggregate classification metrics.

<b><h4>Algorithm Implementation</b></h4>

[Categorical Naive Bayes](../../exaflow/algorithms/exareme3/naive_bayes_categorical.py)

[Gaussian Naive Bayes](../../exaflow/algorithms/exareme3/naive_bayes_gaussian.py)

[Categorical Naive Bayes with Cross - Validation](../../exaflow/algorithms/exareme3/naive_bayes_categorical_cv.py)

[Gaussian Naive Bayes with Cross - Validation](../../exaflow/algorithms/exareme3/naive_bayes_gaussian_cv.py)
