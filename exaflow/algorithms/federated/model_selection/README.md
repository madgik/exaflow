# Model Selection and Cross-Validation

This directory contains federated implementations for model evaluation and selection, including cross-validation infrastructure, data splitting, and scoring functions.

## Table of Contents

1. [Cross Validator](#cross-validator-federatedcrossvalidator) - Model evaluation framework
2. [K-Fold Splitter](#k-fold-splitter-federatedkfoldsplitter) - Data splitting strategy
3. [Scorers](#scorers-classification-regression-multiclass) - Evaluation metrics

---

## Cross Validator (FederatedCrossValidator)

### Name

**Cross Validator (FederatedCrossValidator)**

### Type

**Meta-Estimator** (model evaluation)

### Goal (Why we need it)

Cross-validation evaluates model performance by **training and testing on different data splits**, providing more robust performance estimates than single train/test splits.
In a federated setting, we want the *same* cross-validation metrics as centralized computation **without sharing raw data**, coordinating training/testing across folds using federated aggregation.

### When to use

Use Cross Validator when:

* you need **robust model evaluation** with multiple train/test splits
* you want to tune hyperparameters
* you want to estimate out-of-sample performance
* you have sufficient data for k splits
* you want to use any federated estimator with any splitting strategy

### When NOT to use

Avoid / be careful when:

* data is very small (insufficient for splitting)
* data has temporal dependencies (use time-series splits)
* computational cost is prohibitive (CV is k times slower)

---

### Inputs / Outputs

| Item                 | Description                                                  |
| -------------------- | ------------------------------------------------------------ |
| **estimator**        | FederatedEstimator to evaluate                               |
| **splitter**         | FederatedSplitter defining fold split logic                  |
| **scorer**           | FederatedScorer computing evaluation metrics                 |
| **X**                | Feature matrix (if not using DataFrame API)                  |
| **y**                | Target vector                                                |
| **data**             | Optional DataFrame (alternative to X)                        |
| **categorical_vars** | List of categorical variable names (with DataFrame API)      |
| **numerical_vars**   | List of numerical variable names (with DataFrame API)        |
| **p**                | Number of features (for adjusted metrics)                    |
| **agg_client**       | Federated aggregation client                                 |

**Outputs (dict)**

* `metrics_per_fold`: dict mapping metric names to lists of values (one per fold)
* Common metrics include: accuracy, precision, recall, f1, mse, r2, etc. (depends on scorer)

### Key Differences from scikit-learn

| Aspect           | sklearn cross_validate    | MIP Federated Implementation  |
| ---------------- | ------------------------- | ----------------------------- |
| Data access      | Full centralized data     | Data remains local per client |
| Estimator types  | Any sklearn estimator     | Only federated estimators     |
| Scorer types     | Any scorer                | Only federated scorers        |
| Splitter types   | Any splitter              | Only federated splitters      |
| Performance      | CPU/memory bound          | Network + aggregation bound   |

### Approximation vs Exactness

| Component         | sklearn | MIP                           |
| ----------------- | ------- | ----------------------------- |
| Split logic       | Exact   | Exact                         |
| Estimator fitting | Exact   | Exact (per algorithm)         |
| Scoring           | Exact   | Exact (per metric)            |
| Aggregation       | N/A     | Exact (metrics per fold)      |

---

## K-Fold Splitter (FederatedKFoldSplitter)

### Name

**K-Fold Splitter (FederatedKFoldSplitter)**

### Type

**Data Splitter** (cross-validation utility)

### Goal (Why we need it)

K-Fold splitting divides data into **k equal-sized folds**, using k-1 folds for training and 1 for testing in each iteration.
In a federated setting, splits are computed locally on each client separately, ensuring deterministic fold assignment without data sharing.

### When to use

Use K-Fold Splitter when:

* you want **standard k-fold cross-validation**
* data is i.i.d. (independent and identically distributed)
* no temporal or spatial dependencies exist
* you want reproducible splits (via random_state)

### When NOT to use

Avoid / be careful when:

* data has temporal order (use time-series or stratified splits)
* class imbalance is severe (use stratified k-fold)
* data size is not divisible by k (some folds will differ slightly in size)

---

### Inputs / Outputs

| Item             | Description                                              |
| ---------------- | -------------------------------------------------------- |
| **n_splits**     | Number of folds (k)                                      |
| **shuffle**      | Whether to shuffle before splitting (default: False)     |
| **random_state** | Random seed for reproducibility (if shuffle=True)        |

**Outputs**

* `split(X, y)`: yields (X_train, y_train, X_test, y_test) for each fold
* `split_indices(n)`: yields (train_indices, test_indices) for each fold

### Key Differences from scikit-learn

| Aspect      | sklearn KFold         | MIP Federated Implementation |
| ----------- | --------------------- | ---------------------------- |
| Split logic | Same                  | Same                         |
| Local/Global| Centralized splitting | Local splitting per client   |

### Approximation vs Exactness

| Component   | sklearn | MIP   |
| ----------- | ------- | ----- |
| Fold splits | Exact   | Exact |

---

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
* you want standard metrics (accuracy, precision, recall, F1, MSE, R², etc.)
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
* `r2_adjusted`: adjusted R² (accounting for p features)

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
