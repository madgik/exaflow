## Scorers (Classification, Regression, Multiclass)

### Name

**Scorers (FederatedScorerClassification, FederatedScorerRegression, FederatedScorerMulticlass)**

### Type

**Evaluation Metrics** (model scoring)

### Goal (Why we need it)

Scorers compute **performance metrics** for trained models on test data.
In a federated setting, metrics are computed using federated aggregation of confusion matrix elements (classification) or sums of residuals (regression) **without sharing raw predictions**.

### When to use

Use Scorers when:

* you need to evaluate model performance
* you want standard metrics (accuracy, precision, recall, F1, MSE, RΒ², etc.)
* you're using cross-validation or model selection

**Classification Scorer**: Use for binary classification
**Regression Scorer**: Use for continuous outcomes
**Multiclass Scorer**: Use for multi-class classification

### When NOT to use

* When you need custom metrics not provided by these scorers

---

### Inputs / Outputs

**Classification Scorer**

| Item           | Description                                    |
| -------------- | ---------------------------------------------- |
| **results**    | Fitted model results object                    |
| **X_test**     | Test feature matrix                            |
| **y_test**     | Test target vector                             |
| **agg_client** | Federated aggregation client                   |
| **n_train**    | Number of training observations                |
| **p**          | Number of features                             |

**Outputs (dict)**

* `accuracy`: proportion of correct predictions
* `precision`: TP / (TP + FP)
* `recall`: TP / (TP + FN)
* `f1`: harmonic mean of precision and recall
* `specificity`: TN / (TN + FP)
* `auc`: area under ROC curve (if available)

**Regression Scorer**

**Outputs (dict)**

* `mse`: mean squared error
* `rmse`: root mean squared error
* `mae`: mean absolute error
* `r2`: coefficient of determination
* `r2_adjusted`: adjusted RΒ² (accounting for p features)

**Multiclass Scorer**

**Outputs (dict)**

* `accuracy`: proportion of correct predictions
* `macro_precision`: average precision across classes
* `macro_recall`: average recall across classes
* `macro_f1`: average F1 across classes

### Key Differences from scikit-learn

| Aspect            | sklearn metrics       | MIP Federated Implementation  |
| ----------------- | --------------------- | ----------------------------- |
| Data access       | Full centralized data | Data remains local per client |
| Metric computation| Local computation     | Federated aggregation         |
| Available metrics | Extensive library     | Core metrics only             |

### Approximation vs Exactness

| Component      | sklearn | MIP   |
| -------------- | ------- | ----- |
| All metrics    | Exact   | Exact |
| Confusion matrix elements | Exact | Exact |
| Aggregated sums/counts | Exact | Exact |

