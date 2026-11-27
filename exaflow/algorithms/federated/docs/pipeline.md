# Pipeline composition

Implementation: [pipeline.py](../pipeline.py).

`FederatedPipeline(steps)` composes named transformer steps and an optional final
estimator. `fit(agg_client=..., data=..., y=None, categorical_vars=..., numerical_vars=None)` fits transformers on the dataframe. With an estimator as
the final step it transforms the data and returns the estimator's fitted
results; a transformer-only or empty pipeline returns the pipeline itself.

`transform(data, *, categorical_vars, numerical_vars=None)` applies transformer
steps, excludes a final estimator, and requires a NumPy array result.
`fit_transform` supports transformer-only fitting and transformation.
`get_feature_names_out` delegates to the last transformer.

### Federated flow

```text
Validate dataframe, target length, and unique nonempty step names.
Fit each transformer on the supplied dataframe with the shared column lists.
When the final step is an estimator:
    require y
    transform through the fitted transformers
    fit the estimator on the resulting numeric array
Return the fitted estimator results or the transformer-only pipeline.
```

Nonfinal steps must implement the transformer protocol. During transformation,
intermediate values must remain dataframes until the final transformer; the
final value must be an array. Component type/output failures have separate
structured reasons. This implementation does not provide arbitrary scikit-learn
parameter search, cloning, or automatic dataframe schema inference.

Use [column transformation](column_transformer.md), [one-hot encoding](one_hot_encoder.md),
and [passthrough](passthrough.md) components as appropriate. In
[cross-validation](cross_validator.md), refit the transformation on each
training fold to avoid using held-out category information.
