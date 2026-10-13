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
            }

        y_pred = np.asarray(results.predict(X_test), dtype=float).reshape(-1)
        resid_local = y_test - y_pred
        return {
            "rss": float(np.dot(resid_local, resid_local)),
            "sum_abs_resid": float(np.abs(resid_local).sum()),
            "n_test": float(y_test.shape[0]),
            "sum_y": float(y_test.sum()),
            "sum_y_sq": float((y_test**2).sum()),
        }

    def aggregate(
        self,
        local_stats: dict,
        agg_client,
        n_train: int,
        p: int,
    ) -> dict:
        rss_arr = agg_client.sum(np.array([local_stats["rss"]], dtype=float))
        sum_abs_resid_arr = agg_client.sum(
            np.array([local_stats["sum_abs_resid"]], dtype=float)
        )
        n_test_arr = agg_client.sum(np.array([local_stats["n_test"]], dtype=float))
        sum_y_arr = agg_client.sum(np.array([local_stats["sum_y"]], dtype=float))
        sum_y_sq_arr = agg_client.sum(np.array([local_stats["sum_y_sq"]], dtype=float))

        rss = float(np.asarray(rss_arr, dtype=float).reshape(-1)[0])
        sum_abs_resid = float(np.asarray(sum_abs_resid_arr, dtype=float).reshape(-1)[0])
        n_test = int(np.asarray(n_test_arr, dtype=float).reshape(-1)[0])
        sum_y = float(np.asarray(sum_y_arr, dtype=float).reshape(-1)[0])
        sum_y_sq = float(np.asarray(sum_y_sq_arr, dtype=float).reshape(-1)[0])
        if n_test == 0:
            raise ValueError("Regression scoring requires a nonempty global test set.")

        mean_correction = sum_y * (sum_y / n_test)
        tss = sum_y_sq - mean_correction
        # The subtraction can leave a small signed residual for constant outcomes.
        tss_tol = 100.0 * np.finfo(float).eps * max(abs(sum_y_sq), abs(mean_correction))
        if tss < -tss_tol:
            raise ValueError("Aggregated test outcome variance is negative.")
        has_variation = tss > tss_tol

        rmse_val = float(np.sqrt(rss / n_test))
        mae_val = float(sum_abs_resid / n_test)
        r2_val = (
            float(1.0 - rss / tss) if n_test >= 2 and has_variation else float("nan")
        )

        df_resid = n_train - p - 1
        f_val = float("nan")
        if p > 0 and df_resid > 0 and has_variation:
            if rss <= tss * np.finfo(float).eps * 100.0:
                f_val = float("inf")
            else:
                # CV diagnostic: held-out sums of squares, training degrees of freedom.
                f_val = float((tss - rss) * df_resid / (p * rss))

        return {"rmse": rmse_val, "r2": r2_val, "mae": mae_val, "f_stat": f_val}
