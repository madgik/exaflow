<b><h2><center>Linear Regression</center></h1></b>

<b><h4>Aggregation Server</h4></b>
Some algorithms use the aggregation server to combine partial vectors (e.g., sums)
from workers into a single global result. The controller coordinates the flow and
workers send partial aggregates to the aggregation server via gRPC; the combined
result is then used in the algorithm’s global step.

<b><h4> Notation </h4></b>
Each local dataset *D<sup>(l)</sup>*, where *l*=1,...,*L*, is represented as a matrix of size *n* x *p*, where *L* is the number of medical centers, *n* is the number of points (patients) and *p* is the number of attributes. The elements of the above matrix can either be continuous or discrete (categorical).

In each local dataset, the independent attributes are denoted as a matrix *X<sup>(l)</sup>* and the dependent variable is denoted as a vector *y<sup>(l)</sup>*. *x*<sub>(*ij*)</sub><sup>(*l*)</sup> is the value of the *i*<sup>(*th*)</sup> patient of the *j*<sup>(*th*)</sup> attribute in the *l*<sup>(*th*)</sup> hospital, while *x*<sub>(*j*)</sub><sup>(*l*)</sup> denotes the vector of the *j*<sup>(*th*)</sup> attribute in the *l*<sup>(*th*)</sup> hospital. For categorical attributes, we use the notation *C*<sub>m</sub> <img src="https://render.githubusercontent.com/render/math?math=\epsilon"> { *C*<sub>1</sub>, *C*<sub>2</sub>, ..., *C*<sub>M</sub>} for their domain.

<b><h4> Algorithm Description </h4></b>
Linear regression is a linear approach to modeling the relationship between a dependent variable and one or more independent variables. Here, _y_ should be numerical while _X_ should be continuous or categorical.

![pseudo](images/linear_reg_pseudocode.png)

Once the process has been completed we compute the usual diagnostics as follows.
The local workers compute and broadcast to the central worker the quantities min(ε<sub>i</sub>), max(ε<sub>i</sub>), sum(ε<sub>i</sub>), max(ε<sub>i</sub><sup>2</sup>), where ε<sub>i</sub> are the residuals, as well as the partial *SST* and *SSE*. The central worker then integrates these values to compute the corresponding global ones.
From these quantities the central worker then computes the following diagnostic quantities:

1. For each coefficient β<sub>k</sub>, the *SE*, *t*-statistic and Pr(>|t|)
1. min, max, mean and SE of residuals ε<sub>i</sub> and the degrees of freedom
1. R^2 and Adjusted R^2
1. *F*-statistic and *p*-value

<b><h4>Exareme3 Notes</h4></b>

- Categorical predictors are one-hot encoded via a federated column transformer.
- The intercept term is handled inside the estimator (`fit_intercept=True`).
- Output summary includes log-likelihood, AIC and BIC in addition to standard OLS stats.
- Feature names include an explicit `"Intercept"` entry followed by the expanded columns.

<b><h4>Algorithm Implementation</b></h4>

[Linear Regression](../../exaflow/algorithms/exareme3/linear_regression.py)

[Linear Regression with Cross - Validation](../../exaflow/algorithms/exareme3/linear_regression_cv.py)
