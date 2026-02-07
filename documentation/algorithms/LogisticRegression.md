## Logistic Regression

<b><h4>Aggregation Server</h4></b>
Some algorithms use the aggregation server to combine partial vectors (e.g., sums)
from workers into a single global result. The controller coordinates the flow and
workers send partial aggregates to the aggregation server via gRPC; the combined
result is then used in the algorithm’s global step.

#### Algorithm Description

Logistic Regression training is done by Maximum Likelihood Estimation (MLE) by gradient
descent using, for example, Newton's method. Applying Newton's method leads to the
following algorithm, called __Iteratively Reweighted Least Squares__ (IRLS). Here
the dependent variable *y* has to be binary.

![pseudo](images/logistic_regression_pseudocode.png)

#### Exareme3 Notes

- Categorical predictors are one-hot encoded via a federated column transformer.
- The intercept term is handled inside the estimator (`fit_intercept=True`).
- `positive_class` determines which label is treated as the positive class.
- The summary includes coefficients, standard errors, Wald z-scores, p-values,
  confidence intervals, McFadden/Cox–Snell R², and AIC/BIC.
- CV reports per-fold metrics, confusion matrix, and ROC/AUC curves.

<b><h4>Algorithm Implementation</b></h4>

[Logistic Regression](../../exaflow/algorithms/exareme3/logistic_regression.py)

[Logistic Regression with Cross - Validation](../../exaflow/algorithms/exareme3/logistic_regression_cv.py)
