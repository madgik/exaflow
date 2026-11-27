# Categorical Naive Bayes

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

## Method

The mathematical method, library API, aggregation flow, numerical decisions, reference comparisons, and limitations are documented in [Categorical Naive Bayes](../../exaflow/algorithms/federated/docs/categorical_naive_bayes.md).

## Application integration

The wrapper ordinal-encodes categorical features with aligned categories before fitting and maps library labels and parameters into the response fields below.

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
