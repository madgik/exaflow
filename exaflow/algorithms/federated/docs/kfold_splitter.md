# K-fold splitting

Implementation: [model_selection/cross_validation/splitter_kfold.py](../model_selection/cross_validation/splitter_kfold.py).

`FederatedKFoldSplitter(n_splits, *, shuffle=False, random_state=None)` uses
scikit-learn's `KFold` separately on each participant's local rows. The default
preserves local row order. With shuffling, use a valid shared seed for
reproducibility.

`split(X, y)` yields `(X_train, y_train, X_test, y_test)`.
`split_indices(n_rows)` yields `(train_indices, test_indices)`. Input row
counts must match, `n_splits >= 2`, and every local partition must contain at
least `n_splits` rows; otherwise `INSUFFICIENT_FOLD_SAMPLES` is raised.
Invalid effective splitter settings use `KFOLD_SETTINGS_INVALID`.

## Behavior and limitations

No aggregation is required for splitting. Corresponding local fold indices
are combined by the [cross-validator](cross_validator.md), not constructed
from one global concatenated row order. Centralized parity therefore requires
matching that distributed fold membership.

Folds can differ by one row when counts are not divisible by `n_splits`.
Splits are not stratified, group-aware, or time-aware. The array iterator uses
reusable buffers; consume a fold before advancing or copy it if retaining it.
