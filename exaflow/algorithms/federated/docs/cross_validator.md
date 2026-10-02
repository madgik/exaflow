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

- you need **robust model evaluation** with multiple train/test splits
- you want to tune hyperparameters
- you want to estimate out-of-sample performance
- you have sufficient data for k splits
- you want to use any federated estimator with any splitting strategy

### When NOT to use

Avoid / be careful when:

- data is very small (insufficient for splitting)
- data has temporal dependencies (use time-series splits)
- computational cost is prohibitive (CV is k times slower)

______________________________________________________________________

### Inputs / Outputs

| Item | Description |
| -------------------- | ------------------------------------------------------------ |
| **estimator** | FederatedEstimator to evaluate |
| **splitter** | FederatedSplitter defining fold split logic |
| **scorer** | FederatedScorer computing evaluation metrics |
| **X** | Feature matrix (if not using DataFrame API) |
| **y** | Target vector |
| **data** | Optional DataFrame (alternative to X) |
| **categorical_vars** | List of categorical variable names (with DataFrame API) |
| **numerical_vars** | List of numerical variable names (with DataFrame API) |
| **agg_client** | Federated aggregation client |

**Outputs (dict)**

- `metrics_per_fold`: dict mapping metric names to lists of values (one per fold)
- Regression metrics are `rmse`, `mae`, `r2`, and `f_stat`, plus `n_obs`
  (global training rows per fold). Classification outputs depend on the scorer.

The predictor count passed to scorers is derived from each fold's transformed
feature matrix, excluding the intercept added by the estimator. `evaluate`
does not accept a predictor-count override. Counts must not be summed across
workers sharing the same feature schema.

Regression metrics are calculated independently. Undefined R-squared or F
does not erase valid errors. Nonfinite fold metrics are retained internally;
the linear regression CV response summarizes an affected metric as
`[null, null]`. See [linear regression CV](../../../../documentation/algorithms/linear_regression_cv.md)
for formulas, edge cases, and the `root_mean_sq_error` response contract.

### Key Differences from scikit-learn

| Aspect | sklearn cross_validate | MIP Federated Implementation |
| ---------------- | ------------------------- | ----------------------------- |
| Data access | Full centralized data | Data remains local per client |
| Estimator types | Any sklearn estimator | Only federated estimators |
| Scorer types | Any scorer | Only federated scorers |
| Splitter types | Any splitter | Only federated splitters |
| Performance | CPU/memory bound | Network + aggregation bound |

### Approximation vs Exactness

| Component | sklearn | MIP |
| ----------------- | ------- | ----------------------------- |
| Split logic | Global KFold | Local KFold on each worker |
| Estimator fitting | Exact | Exact (per algorithm) |
| Scoring | Exact | Exact (per metric) |
| Aggregation | N/A | Exact (metrics per fold) |

Numerical parity requires the centralized reference to use the same local
fold membership and aligned feature schemas. Concatenating all rows and
applying KFold usually produces different folds.

______________________________________________________________________
