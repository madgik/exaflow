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


