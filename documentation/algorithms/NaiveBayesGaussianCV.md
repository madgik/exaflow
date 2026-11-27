# Gaussian Naive Bayes K-fold Cross-validation

Runtime wrapper: [naive_bayes/naive_bayes_gaussian_cv.py](../../exaflow/algorithms/exareme3/naive_bayes/naive_bayes_gaussian_cv.py).

CV variants stay beside their models in the same runtime domain.

## Overview

This algorithm evaluates Gaussian Naive Bayes with K-fold cross-validation and
returns a multiclass confusion matrix plus fold-level classification metrics.

## Inputs

### Required inputs

| Input | Description |
|---|---|
| `y` | Nominal class label. |
| `x` | Numerical features. |

### Parameters

| Parameter | Description | Default |
|---|---|---|
| `n_splits` | Number of cross-validation folds. | `5` |

## Method

See the [library method](../../exaflow/algorithms/federated/docs/gaussian_naive_bayes.md) and numerical input preparation for the statistical calculation and core contracts.

Shared [cross-validation](../../exaflow/algorithms/federated/docs/cross_validator.md), [K-fold splitting](../../exaflow/algorithms/federated/docs/kfold_splitter.md), and [scoring](../../exaflow/algorithms/federated/docs/scorers.md) describe fold execution and metrics.

## Application integration

The wrapper uses the requested fold count (default 5), runs the required missing-value preprocessing, and summarizes the held-out fold results. It does not return a final model refitted on all observations.

## Outputs

| Field | Description |
|---|---|
| `confusion_matrix` | Aggregated multiclass confusion matrix. |
| `classification_summary` | Accuracy, precision, recall, F-score, and observation counts by fold and class. |
