# Gaussian Naive Bayes

Implementation: [naive_bayes/naive_bayes_gaussian.py](../naive_bayes/naive_bayes_gaussian.py).

Gaussian Naive Bayes classifies observations with numerical features by modeling
each feature as Gaussian within each class.

## Library interface

`FederatedGaussianNB(x_vars=..., labels=..., var_smoothing=1e-9).fit(X, y, *, agg_client)` takes numerical features and aligned class labels. The result exposes `theta`, `var`, `class_count`, `class_prior`, `nobs`, and prediction methods. Variance smoothing is configurable at the library level.

## Statistical method

For class `c` and feature `j`:

```text
x_j | y = c ~ Normal(theta_cj, var_cj)
```

Prediction chooses the class with largest posterior probability:

```text
argmax_c P(y = c) product_j P(x_j | y = c)
```

## Federated computation

The model is fitted without sharing row-level data. Each site contributes
class-feature sufficient statistics.

### Aggregated quantities

| Quantity | Purpose |
|---|---|
| Class labels | Align class rows. |
| Class-feature counts | Compute class counts and handle missing feature values. |
| Class-feature sums | Compute class-conditional means. |
| Class-feature sums of squares | Compute class-conditional variances. |

### Federated flow

```text
Input:
    y: class labels
    X: numerical features

Step 1:
    Determine the global class labels.

Step 2:
    At each site, for each class and feature:
        count non-missing values
        compute sum
        compute sum of squares

Step 3:
    Aggregate counts, sums, and sums of squares.

Step 4:
    Compute class means, variances, class counts, and class priors.

Step 5:
    For prediction:
        evaluate Gaussian likelihoods
        multiply by class priors
        normalize posterior probabilities
        choose the largest posterior

Output:
    fitted Gaussian Naive Bayes parameters
```

## Numerical decisions

- Feature values are coerced to numerical arrays.
- Class priors are estimated from aggregated class counts.
- Variances use class-conditional population variance.
- Variances are clipped by `var_smoothing * max_variance`; if this is invalid,
  `var_smoothing` is used directly.
- The implementation default for `var_smoothing` is `1e-9`.

## Reference comparison

Standalone tests compare behavior with:

```text
sklearn.naive_bayes.GaussianNB()
```

Reference behavior is aligned with scikit-learn Gaussian Naive Bayes, with
parameters computed from aggregated class-feature statistics.

## Limitations

- Features must be numerical.
- The method assumes conditional independence of features given the class.
- Each feature is modeled as Gaussian within each class.
- Strongly non-Gaussian feature distributions can reduce classifier quality.
