# Gaussian Naive Bayes K-fold Cross-validation

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

## Classification method

Each fold fits the Gaussian Naive Bayes model described in
[Gaussian Naive Bayes](NaiveBayesGaussian.md) on training partitions and
evaluates predictions on held-out partitions.

## Federated computation

Cross-validation is computed without sharing row-level data. Fold-specific model
statistics and evaluation counts are aggregated.

### Aggregated quantities

| Quantity | Purpose |
|---|---|
| Fold train/test assignment summaries | Coordinate K-fold evaluation. |
| Gaussian Naive Bayes sufficient statistics | Fit each fold model. |
| Confusion-matrix counts | Aggregate classification outcomes. |
| Per-class true/false positive/negative counts | Compute accuracy, precision, recall, and F-score. |

### Federated flow

```text
Input:
    y: class labels
    X: numerical features
    n_splits: number of folds

Step 1:
    Build K-fold train/test partitions.

Step 2:
    For each fold:
        fit Gaussian Naive Bayes from aggregated training statistics
        predict held-out observations at each site
        compute local confusion-matrix counts

Step 3:
    Aggregate fold confusion matrices and class-level metric counts.

Output:
    confusion matrix and fold-level classification summary
```

## Technical decisions

- The classifier uses the same Gaussian Naive Bayes fit behavior as the non-CV
  variant.
- Metrics are computed from aggregated confusion counts.
- The default fold count is `5`.

## Outputs

| Field | Description |
|---|---|
| `confusion_matrix` | Aggregated multiclass confusion matrix. |
| `classification_summary` | Accuracy, precision, recall, F-score, and observation counts by fold and class. |

## Validation against state-of-the-art implementation

Classifier behavior is aligned with `sklearn.naive_bayes.GaussianNB`. The
cross-validation output is validated through algorithm validation fixtures and
classification-metric tests.

## Limitations and assumptions

- Features must be numerical.
- All limitations of Gaussian Naive Bayes apply.
- Cross-validation estimates performance for the selected fold split, not a
  final refit on all observations.
