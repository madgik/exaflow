# Categorical Naive Bayes

## Table of contents

- [Overview](#overview)
- [Inputs](#inputs)
  - [Required inputs](#required-inputs)
  - [Parameters](#parameters)
- [Classification method](#classification-method)
- [Federated computation](#federated-computation)
  - [Aggregated quantities](#aggregated-quantities)
  - [Federated flow](#federated-flow)
- [Technical decisions](#technical-decisions)
- [Outputs](#outputs)
- [Validation against state-of-the-art implementation](#validation-against-state-of-the-art-implementation)
- [Limitations and assumptions](#limitations-and-assumptions)

## Overview

Categorical Naive Bayes classifies observations with nominal features by
estimating class-conditional category probabilities.

## Inputs

### Required inputs

| Input | Description |
|---|---|
| `y` | Nominal class label. |
| `x` | Nominal features. |

### Parameters

No user parameters are exposed for the fit/predict variant.

## Classification method

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
Input:
    y: class labels
    X: categorical features

Step 1:
    Determine global class labels.

Step 2:
    Ordinal-encode feature categories with aligned category order.

Step 3:
    At each site:
        count observations per class
        count each feature category within each class

Step 4:
    Aggregate class counts and class-feature-category counts.

Step 5:
    Apply additive smoothing to category probabilities.

Step 6:
    For prediction:
        compute class priors
        multiply class-conditional category probabilities
        ignore unknown category codes by assigning factor 1
        choose the largest posterior

Output:
    fitted categorical Naive Bayes parameters
```

## Technical decisions

- Additive smoothing uses `alpha = 1.0`.
- Unknown encoded categories are represented as `-1` and ignored during
  prediction.
- Class labels are sorted after aggregation.
- Feature categories come from the aligned encoder/categories supplied to the
  estimator.
- Probabilities are stored in log form for fitted parameters.

## Outputs

| Field | Description |
|---|---|
| `classes` | Class labels included in the fitted model. |
| `class_count` | Number of observations per class. |
| `category_count` | Category counts for each feature and class. |
| `class_log_prior` | Log class prior probabilities. |
| `category_log_prob` | Log class-conditional category probabilities. |
| `categories` | Feature categories in model order. |
| `feature_names` | Feature names in model order. |

## Validation against state-of-the-art implementation

Standalone tests compare behavior with:

```text
sklearn.naive_bayes.CategoricalNB(alpha=1.0)
```

Reference behavior is aligned with scikit-learn categorical Naive Bayes using
additive smoothing.

## Limitations and assumptions

- Features must be categorical.
- The method assumes conditional independence of features given the class.
- Unknown categories at prediction time are ignored rather than assigned a
  learned probability.
- Category ordering must remain aligned between fitting and prediction.
