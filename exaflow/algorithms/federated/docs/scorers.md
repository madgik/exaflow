# Federated scorers

Implementations: [regression](../model_selection/cross_validation/scorer_regression.py),
[binary classification](../model_selection/cross_validation/scorer_classification.py),
and [multiclass classification](../model_selection/cross_validation/scorer_multiclass.py).

Each scorer exposes `score(results, X_test, y_test, *, agg_client, n_train, p)`,
with separate `local` and `aggregate` methods. Predictions remain local; only
residual statistics or confusion counts are summed. `results` must support
prediction in the format expected by the selected scorer.

## Regression

`FederatedRegressionScorer` returns `rmse`, `r2`, `mae`, and `f_stat`.
Let `p` be encoded predictor width excluding an estimator-added intercept;
`n_train` and `n_test` are global training and test counts.

```text
RSS = sum_test((y - prediction)^2)
TSS = sum_test(y^2) - sum_test(y)^2 / n_test
RMSE = sqrt(RSS / n_test)
MAE = sum_test(abs(y - prediction)) / n_test
R_squared = 1 - RSS / TSS
F_diagnostic = (TSS - RSS) * (n_train - p - 1) / (p * RSS)
```

RMSE and MAE have the outcome's units. R-squared and the F diagnostic can be
negative. The F diagnostic combines held-out sums of squares with training
degrees of freedom; it is not a conventional fitted-model F-test and has no
significance p-value.

### Federated flow

```text
Predict on each local test partition.
Compute RSS, absolute residual sum, test count, outcome sum, and squared sum.
Sum each statistic across participants.
Compute TSS and each metric independently using the global counts.
Return raw numeric metric values, including NaN/inf for undefined cases.
```

Zero prediction error remains zero. R-squared is `NaN` with fewer than two test
observations or no numerically resolvable outcome variation. The F diagnostic
is `NaN` when `p=0`, residual degrees of freedom are nonpositive, or TSS has no
resolvable variation; a perfect nonconstant fit gives infinity. Floating-point
tolerances handle cancellation/perfect-fit detection without rounding RMSE or
MAE to zero. Large offsets with small variation can still cause cancellation.
An empty global test set raises `REGRESSION_TEST_SET_EMPTY`; empty local test
partitions contribute zeros. Materially negative computed variance has the
separate `REGRESSION_OUTCOME_VARIANCE_INVALID` diagnostic.

## Binary classification

`FederatedClassificationScorer(thresholds=None)` aggregates `tp`, `fp`, `tn`,
`fn`, and threshold-specific counts. It returns these confusion counts plus
`roc_tpr` and `roc_fpr` lists. Its threshold grid defaults to 101 values between
0 and 1. ROC points are threshold-based, not an exact rank-based AUC.
`compute_classification_metrics_from_confmat` separately derives accuracy,
precision, recall, and F score; zero denominators produce zero.

## Multiclass classification

`FederatedMulticlassClassificationScorer(labels=None)` uses explicit labels or
`results.labels`, aggregates an aligned class-by-class confusion matrix, and
returns `confmat` as nested lists. Missing label configuration is an explicit
failure. Rows with true or predicted labels outside the configured universe
are ignored. `multiclass_classification_metrics` separately computes per-class
accuracy, precision, recall, and F score; its summary helper combines fold
metrics. Neither scorer silently adds regression metrics or adjusted R-squared.

## Reference comparisons

Regression tests compare held-out metrics with centralized calculations using
the same predictions and folds, and check the retained F formula separately.
Confusion counts are exact for aligned labels and thresholds; derived metrics
follow their documented denominator conventions. The
[cross-validator](cross_validator.md) retains per-fold values without imposing
a consuming application's summary or missing-value serialization policy.
