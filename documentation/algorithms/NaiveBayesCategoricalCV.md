# Categorical Naive Bayes K-fold Cross-validation

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

This algorithm evaluates categorical Naive Bayes with K-fold cross-validation and
returns a multiclass confusion matrix plus fold-level classification metrics.

## Inputs

### Required inputs

| Input | Description |
|---|---|
| `y` | Nominal class label. |
| `x` | Nominal features. |

### Parameters

| Parameter | Description | Default |
|---|---|---|
| `n_splits` | Number of cross-validation folds. | `5` |

## Classification method

Each fold fits the categorical Naive Bayes model described in
[Categorical Naive Bayes](NaiveBayesCategorical.md) on training partitions and
evaluates predictions on held-out partitions.

## Federated computation

Cross-validation is computed without sharing row-level data. Fold-specific
category counts and evaluation counts are aggregated.

### Aggregated quantities

| Quantity | Purpose |
|---|---|
| Fold train/test assignment summaries | Coordinate K-fold evaluation. |
| Class and category counts | Fit each fold model. |
| Confusion-matrix counts | Aggregate classification outcomes. |
| Per-class true/false positive/negative counts | Compute accuracy, precision, recall, and F-score. |

### Federated flow

```text
Input:
    y: class labels
    X: categorical features
    n_splits: number of folds

Step 1:
    Build K-fold train/test partitions.

Step 2:
    For each fold:
        fit categorical Naive Bayes from aggregated training counts
        predict held-out observations at each site
        compute local confusion-matrix counts

Step 3:
    Aggregate fold confusion matrices and class-level metric counts.

Output:
    confusion matrix and fold-level classification summary
```

## Technical decisions

- The classifier uses the same additive-smoothed category-count behavior as the
  non-CV variant.
- Metrics are computed from aggregated confusion counts.
- The default fold count is `5`.

## Outputs

| Field | Description |
|---|---|
| `confusion_matrix` | Aggregated multiclass confusion matrix. |
| `classification_summary` | Accuracy, precision, recall, F-score, and observation counts by fold and class. |

## Validation against state-of-the-art implementation

Classifier behavior is aligned with `sklearn.naive_bayes.CategoricalNB` using
`alpha=1.0`. Cross-validation output is validated through algorithm validation
fixtures and classification-metric tests.

## Limitations and assumptions

- Features must be categorical.
- All limitations of categorical Naive Bayes apply.
- Cross-validation estimates performance for the selected fold split, not a
  final refit on all observations.
