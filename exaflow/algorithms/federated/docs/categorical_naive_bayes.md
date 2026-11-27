# Categorical Naive Bayes

Implementation: [naive_bayes/naive_bayes_categorical.py](../naive_bayes/naive_bayes_categorical.py).

Categorical Naive Bayes classifies observations with nominal features by
estimating class-conditional category probabilities.

## Library interface

`FederatedCategoricalNB(y_var=..., x_vars=..., categories=...).fit(X, y, *, agg_client)` accepts ordinal-encoded features with aligned category lists. The result exposes `labels`, `class_count`, `category_count`, `class_log_prior`, `category_log_prob`, `nobs`, and prediction methods. Use the separate [ordinal encoder](ordinal_encoder.md) to prepare categories.

## Statistical method

For each class `c` and feature category `v`, the model estimates:

```text
P(x_j = v | y = c)
```

Prediction chooses:

```text
argmax_c P(y = c) product_j P(x_j | y = c)
```

## Federated computation

The model is fitted without sharing row-level data. Features are ordinal-encoded
with aligned category order, and each site contributes class/category counts.

### Aggregated quantities

| Quantity | Purpose |
|---|---|
| Class labels | Align class rows. |
| Feature categories | Align category columns for each feature. |
| Class counts | Estimate class priors. |
| Class-feature-category counts | Estimate class-conditional likelihoods. |

### Federated flow

```text
Receive ordinal-encoded X, y, and aligned category lists.
Discover and align global class labels.
Count local observations per class and feature/category within each class.
Sum class and category counts.
Apply additive smoothing (alpha = 1) and compute log probabilities.
At prediction time, ignore unknown code -1 and choose the largest posterior.
```

## Numerical decisions

- Additive smoothing uses `alpha = 1.0`.
- Unknown encoded categories are represented as `-1` and ignored during
  prediction.
- Class labels are sorted after aggregation.
- Feature categories come from the aligned encoder/categories supplied to the
  estimator.
- Probabilities are stored in log form for fitted parameters.

## Reference comparison

Standalone tests compare behavior with:

```text
sklearn.naive_bayes.CategoricalNB(alpha=1.0)
```

Reference behavior is aligned with scikit-learn categorical Naive Bayes using
additive smoothing.

## Limitations

- Features must be categorical.
- The method assumes conditional independence of features given the class.
- Unknown categories at prediction time are ignored rather than assigned a
  learned probability.
- Category ordering must remain aligned between fitting and prediction.
