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


