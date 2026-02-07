from __future__ import annotations

import numpy as np

from exaflow.algorithms.federated.utils.agg_client import AggregationClient
from exaflow.algorithms.federated.utils.interfaces import FederatedEstimatorResults
from exaflow.algorithms.federated.utils.interfaces import FederatedScorer


class FederatedClassificationScorer(FederatedScorer):
    """Federated scorer for classification metrics and ROC curves."""

    def __init__(self, *, thresholds: np.ndarray | None = None) -> None:
        if thresholds is None:
            thresholds = np.linspace(0.0, 1.0, 101)
        self.thresholds = np.asarray(thresholds, dtype=float)

    def local(
        self,
        results: FederatedEstimatorResults,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict:
        y_true = np.asarray(y_test, dtype=float).reshape(-1)
        proba = np.asarray(results.predict(X_test), dtype=float).reshape(-1)

        preds05 = (proba >= 0.5).astype(int)
        tp = int(((preds05 == 1) & (y_true == 1)).sum())
        fp = int(((preds05 == 1) & (y_true == 0)).sum())
        tn = int(((preds05 == 0) & (y_true == 0)).sum())
        fn = int(((preds05 == 0) & (y_true == 1)).sum())

        tp_buf = np.empty_like(self.thresholds, dtype=float)
        fp_buf = np.empty_like(self.thresholds, dtype=float)
        tn_buf = np.empty_like(self.thresholds, dtype=float)
        fn_buf = np.empty_like(self.thresholds, dtype=float)

        for i, thr in enumerate(self.thresholds):
            preds_thr = proba >= thr
            tp_buf[i] = float(((preds_thr) & (y_true == 1)).sum())
            fp_buf[i] = float(((preds_thr) & (y_true == 0)).sum())
            tn_buf[i] = float((~preds_thr & (y_true == 0)).sum())
            fn_buf[i] = float((~preds_thr & (y_true == 1)).sum())

        return {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "tp_list": tp_buf,
            "fp_list": fp_buf,
            "tn_list": tn_buf,
            "fn_list": fn_buf,
        }

    def aggregate(
        self,
        local_stats: dict,
        *,
        agg_client: AggregationClient,
        n_train: int,
        p: int,
    ) -> dict:
        tp_global = int(agg_client.sum(np.array([float(local_stats["tp"])]))[0])
        fp_global = int(agg_client.sum(np.array([float(local_stats["fp"])]))[0])
        tn_global = int(agg_client.sum(np.array([float(local_stats["tn"])]))[0])
        fn_global = int(agg_client.sum(np.array([float(local_stats["fn"])]))[0])

        tp_arr = np.asarray(
            agg_client.sum(np.asarray(local_stats["tp_list"], dtype=float)),
            dtype=float,
        )
        fp_arr = np.asarray(
            agg_client.sum(np.asarray(local_stats["fp_list"], dtype=float)),
            dtype=float,
        )
        tn_arr = np.asarray(
            agg_client.sum(np.asarray(local_stats["tn_list"], dtype=float)),
            dtype=float,
        )
        fn_arr = np.asarray(
            agg_client.sum(np.asarray(local_stats["fn_list"], dtype=float)),
            dtype=float,
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            tpr = np.divide(
                tp_arr,
                tp_arr + fn_arr,
                out=np.zeros_like(tp_arr),
                where=(tp_arr + fn_arr) > 0,
            )
            fpr = np.divide(
                fp_arr,
                fp_arr + tn_arr,
                out=np.zeros_like(fp_arr),
                where=(fp_arr + tn_arr) > 0,
            )

        return {
            "tp": tp_global,
            "fp": fp_global,
            "tn": tn_global,
            "fn": fn_global,
            "roc_tpr": tpr.tolist(),
            "roc_fpr": fpr.tolist(),
        }


def compute_classification_metrics_from_confmat(confmat: dict) -> dict:
    tp = int(confmat.get("tp", 0))
    fp = int(confmat.get("fp", 0))
    tn = int(confmat.get("tn", 0))
    fn = int(confmat.get("fn", 0))

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0

    prec_den = tp + fp
    precision = tp / prec_den if prec_den > 0 else 0.0

    rec_den = tp + fn
    recall = tp / rec_den if rec_den > 0 else 0.0

    if precision + recall > 0:
        fscore = 2.0 * precision * recall / (precision + recall)
    else:
        fscore = 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "fscore": float(fscore),
    }
