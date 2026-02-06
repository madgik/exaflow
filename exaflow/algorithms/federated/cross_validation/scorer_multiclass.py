from __future__ import annotations

import warnings
from typing import List
from typing import Optional

import numpy as np
import pandas as pd

from exaflow.algorithms.federated.agg_client import AggregationClient
from exaflow.algorithms.federated.interfaces import FederatedEstimatorResults
from exaflow.algorithms.federated.interfaces import FederatedScorer


class FederatedMulticlassClassificationScorer(FederatedScorer):
    """Federated scorer for multiclass confusion matrices."""

    def __init__(self, *, labels: Optional[List[str]] = None) -> None:
        self.labels = list(labels) if labels is not None else None

    def local(
        self,
        results: FederatedEstimatorResults,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict:
        labels = self.labels or getattr(results, "labels", None)
        if labels is None:
            raise ValueError(
                "Labels must be provided or available on estimator results."
            )
        label_to_idx = {label: idx for idx, label in enumerate(labels)}

        y_true = np.asarray(y_test)
        y_pred = np.asarray(results.predict(X_test))

        n_classes = len(labels)
        confmat_local = np.zeros((n_classes, n_classes), dtype=float)

        if y_true.size == 0 or y_pred.size == 0:
            return {"confmat": confmat_local}

        true_idx = np.array(
            [label_to_idx.get(label, -1) for label in y_true], dtype=int
        )
        pred_idx = np.array(
            [label_to_idx.get(label, -1) for label in y_pred], dtype=int
        )

        valid_mask = (true_idx >= 0) & (pred_idx >= 0)
        if np.any(valid_mask):
            np.add.at(confmat_local, (true_idx[valid_mask], pred_idx[valid_mask]), 1.0)

        return {"confmat": confmat_local}

    def aggregate(
        self,
        local_stats: dict,
        *,
        agg_client: AggregationClient,
        n_train: int,
        p: int,
    ) -> dict:
        conf_local = np.asarray(local_stats["confmat"], dtype=float)
        conf_global_flat = agg_client.sum(conf_local.ravel())
        conf_global = np.asarray(conf_global_flat, dtype=float).reshape(
            conf_local.shape
        )
        return {"confmat": conf_global.tolist()}


def multiclass_classification_metrics(confmat):
    """
    Computes classification metrics from confusion matrix

    The classification metrics are accuracy, precision, recall and fscore.
    These are all computed starting from a multiclass confusion matrix.

    Parameters
    ----------
    confmat : numpy.array of shape (n_labels, n_labels)
        A multiclass confusion matrix

    Returns
    -------
    dict
        A dictionary containing all the classification metrics
    """
    n_labels, _ = confmat.shape

    # In order to compute the classification metrics we first compute the true
    # positives and negatives, and the false positives and negatives. These are
    # computed by summing the relevand subparts of the confusion matrix,
    # explained below case by case.
    # Then, the classification metrics are computed from the TP, TN, FP, FN.

    # True positives are found in the diagonal of the matrix
    tp = np.diag(confmat)

    # True negatives are the sums of the submatrices, complementary to the
    # diagonal elements
    ix_args = [[[i for i in range(n_labels) if i != j]] * 2 for j in range(n_labels)]
    tn = np.array([confmat[np.ix_(*args)].sum() for args in ix_args])

    # For false negatives we sum every row omitting the diagonal elements
    fn_idcs = [
        ([i] * (n_labels - 1), [j for j in range(n_labels) if j != i])
        for i in range(n_labels)
    ]
    fn = np.array([confmat[idx] for idx in fn_idcs]).sum(axis=1)

    # For false positives we sum every column omitting the diagonal elements, hence
    # we need to swap fn indices
    fp_idcs = [(lambda a, b: (b, a))(*idx) for idx in fn_idcs]
    fp = np.array([confmat[idx] for idx in fp_idcs]).sum(axis=1)

    # Divisions by zero raise warnings but we replace NaNs later anyway
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fscore = 2 * (precision * recall) / (precision + recall)

    # Replace NaNs with 0s
    accuracy = np.nan_to_num(accuracy)
    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    fscore = np.nan_to_num(fscore)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
    }


def multiclass_classification_summary(metrics, labels, n_obs):
    """Formats the classification metrics into a summary table, represented by
    a nested dict."""
    # Zip values with labels and index by fold
    data = {
        f"fold{i}": {k: dict(zip(labels, v)) for k, v in m.items()}
        for i, m in enumerate(metrics)
    }

    # Reformat nested dict in a format understood by pandas as a multi-index.
    reform = {
        fold_key: {
            (metrics_key, level): val
            for metrics_key, metrics_vals in fold_val.items()
            for level, val in metrics_vals.items()
        }
        for fold_key, fold_val in data.items()
    }
    # Then transpose to convert multi-index dataframe into hierarchical one
    # (multi-index on the columns).
    df = pd.DataFrame(reform).T

    # Append rows for average and stdev of every column
    df.loc["average"] = df.mean()
    df.loc["stdev"] = df.std()

    # Hierarchical dataframe to nested dict
    summary = {level: df.xs(level, axis=1).to_dict() for level in df.columns.levels[0]}

    summary["n_obs"] = {f"fold{i}": int(n) for i, n in enumerate(n_obs)}
    return summary
