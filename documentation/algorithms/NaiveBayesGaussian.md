# Gaussian Naive Bayes

## Overview

Gaussian Naive Bayes classifies observations with numerical features by modeling
each feature as Gaussian within each class.

## Inputs

### Required inputs

| Input | Description |
|---|---|
| `y` | Nominal class label. |
| `x` | Numerical features. |

### Parameters

No user parameters are exposed for the fit/predict variant.

## Method

The mathematical method, library API, aggregation flow, numerical decisions, reference comparisons, and limitations are documented in [Gaussian Naive Bayes](../../exaflow/algorithms/federated/docs/gaussian_naive_bayes.md).

## Application integration

The wrapper supplies numerical features and globally aligned class labels, then maps the fitted parameters into the response fields below.

## Outputs

| Field | Description |
|---|---|
| `classes` | Class labels included in the fitted model. |
| `class_count` | Number of observations per class. |
| `theta` | Class-feature means. |
| `var` | Class-feature variances after smoothing. |
| `class_prior` | Estimated class prior probabilities. |
| `feature_names` | Feature names in model order. |
