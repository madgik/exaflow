# Cross-validation

Implementation: [model_selection/cross_validation/cross_validator.py](../model_selection/cross_validation/cross_validator.py).

`FederatedCrossValidator(estimator=..., splitter=..., scorer=...).evaluate(X, y, *, data=None, categorical_vars=None, numerical_vars=None, agg_client=...)`
coordinates fitting and scoring. It returns a dictionary of per-fold metric
lists, including global training `n_obs`; it does not summarize or serialize
those lists for a consuming application.

## Components and inputs

Use the estimator, results, splitter, and scorer protocols from
[interfaces.py](../utils/interfaces.py). Shared implementations include
[K-fold splitting](kfold_splitter.md), [scorers](scorers.md),
[OLS](ols.md), the [Naive Bayes models](../README.md#naive_bayes), and
[pipeline composition](pipeline.md).

The array path accepts `X` and `y`. The dataframe path accepts `data`, `y`,
`categorical_vars`, and `numerical_vars`, requires `split_indices`, and uses
an estimator supporting dataframe fitting and transformation. A fitted
transformer must learn category schemas from the training fold, then reuse
that schema for test data; see [one-hot encoding](one_hot_encoder.md).

### Federated flow

```text
For each corresponding local fold:
    split local features/outcomes into training and test partitions
    fit the estimator using training data and the aggregation client
    transform held-out dataframe features with the fitted training schema
    obtain n_train from results.nobs and p from transformed test width
    score held-out predictions using aggregated statistics
    append every metric and training n_obs to its per-fold list
Return the metric lists without averaging or dropping folds.
```

## Decisions and limitations

Fold membership is determined locally by the splitter. A distributed split
usually differs from splitting concatenated data; centralized comparisons must
use the same membership and encoding. All participants must execute the same
fold sequence and collective operations. Predictor count is an aligned schema
width, not a count summed across participants.

Data/outcome row counts are checked. Missing dataframe feature lists, an
unsupported splitter interface, and absent array inputs have distinct diagnostic
reasons. Numerical scoring edge cases are documented in [scorers](scorers.md).
No final model refit on all rows is performed by `evaluate`.
