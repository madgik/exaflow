from __future__ import annotations

import numpy as np

from exaflow.algorithms.federated.utils.interfaces import FederatedEstimatorResults
from exaflow.algorithms.federated.utils.interfaces import FederatedScorer


class FederatedRegressionScorer(FederatedScorer):
    """Compute regression metrics using federated aggregation."""

    def score(
        self,
        results: FederatedEstimatorResults,
        X_test: np.ndarray,
        y_test: np.ndarray,
        *,
        agg_client,
        n_train: int,
        p: int,
    ) -> dict:
        local_stats = self.local(results, X_test, y_test)
        return self.aggregate(local_stats, agg_client=agg_client, n_train=n_train, p=p)

    def local(
        self,
        results: FederatedEstimatorResults,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict:
        y_test = np.asarray(y_test, dtype=float).reshape(-1)
        if y_test.size == 0:
            return {
                "rss": 0.0,
                "sum_abs_resid": 0.0,
                "n_test": 0.0,
                "sum_y": 0.0,
                "sum_y_sq": 0.0,
                "p": 0.0,
            }

        y_pred = np.asarray(results.predict(X_test), dtype=float).reshape(-1)
        resid_local = y_test - y_pred
        return {
            "rss": float(np.dot(resid_local, resid_local)),
            "sum_abs_resid": float(np.abs(resid_local).sum()),
            "n_test": float(y_test.shape[0]),
            "sum_y": float(y_test.sum()),
            "sum_y_sq": float((y_test**2).sum()),
            "p": float(X_test.shape[1]) if X_test.ndim > 1 else 0.0,
        }

    def aggregate(
        self,
        local_stats: dict,
        agg_client,
        n_train: int,
        p: int | None = None,
    ) -> dict:
        rss_arr = agg_client.sum(np.array([local_stats["rss"]], dtype=float))
        sum_abs_resid_arr = agg_client.sum(
            np.array([local_stats["sum_abs_resid"]], dtype=float)
        )
        n_test_arr = agg_client.sum(np.array([local_stats["n_test"]], dtype=float))
        sum_y_arr = agg_client.sum(np.array([local_stats["sum_y"]], dtype=float))
        sum_y_sq_arr = agg_client.sum(np.array([local_stats["sum_y_sq"]], dtype=float))
        p_arr = agg_client.sum(np.array([local_stats["p"]], dtype=float))

        rss = float(np.asarray(rss_arr, dtype=float).reshape(-1)[0])
        sum_abs_resid = float(np.asarray(sum_abs_resid_arr, dtype=float).reshape(-1)[0])
        n_test = int(np.asarray(n_test_arr, dtype=float).reshape(-1)[0])
        sum_y = float(np.asarray(sum_y_arr, dtype=float).reshape(-1)[0])
        sum_y_sq = float(np.asarray(sum_y_sq_arr, dtype=float).reshape(-1)[0])
        p_val = int(np.asarray(p_arr, dtype=float).reshape(-1)[0])
        if p is None:
            p = p_val

        if n_test > 0:
            y_mean = sum_y / n_test
            tss = sum_y_sq - 2.0 * y_mean * sum_y + n_test * (y_mean**2)
        else:
            tss = 0.0

        df_resid = n_train - p - 1

        if df_resid <= 0 or n_test == 0 or rss <= 0.0 or tss <= 0.0 or p <= 0:
            return {"rmse": 0.0, "r2": 0.0, "mae": 0.0, "f_stat": 0.0}

        r2_val = 1.0 - (rss / tss)
        rmse_val = float(np.sqrt(rss / n_test))
        mae_val = float(sum_abs_resid / n_test)
        f_val = float((tss - rss) * df_resid / (p * rss))

        return {"rmse": rmse_val, "r2": r2_val, "mae": mae_val, "f_stat": f_val}
