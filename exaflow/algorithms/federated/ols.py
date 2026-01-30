from __future__ import annotations

import numpy as np
from scipy import stats

ALPHA = 0.05


class FederatedOLSResults:
    """Container for fitted federated OLS statistics."""

    def __init__(
        self,
        *,
        params,
        bse,
        tvalues,
        pvalues,
        df_resid,
        df_model,
        rse,
        fvalue,
        f_pvalue,
        r_squared,
        r_squared_adjusted,
        n_obs,
        rss,
        tss,
        sum_abs_resid,
        rank_,
        xTx_inv_,
        cov_params,
    ):
        self.params = np.asarray(params, dtype=float)
        self.bse = np.asarray(bse, dtype=float)
        self.tvalues = np.asarray(tvalues, dtype=float)
        self.pvalues = np.asarray(pvalues, dtype=float)
        self.df_resid = float(df_resid)
        self.df_model = float(df_model)
        self.rse = float(rse)
        self.fvalue = float(fvalue)
        self.f_pvalue = float(f_pvalue)
        self.rsquared = float(r_squared)
        self.rsquared_adj = float(r_squared_adjusted)
        self.nobs = int(n_obs)
        self.rss = float(rss)
        self.tss = float(tss)
        self.sum_abs_resid = float(sum_abs_resid)
        self.rank_ = int(rank_)
        self.xTx_inv_ = np.asarray(xTx_inv_, dtype=float)
        self.cov_params = np.asarray(cov_params, dtype=float)

    @property
    def coefficients(self):
        return self.params

    @property
    def std_err(self):
        return self.bse

    @property
    def t_stats(self):
        return self.tvalues

    def conf_int(self, alpha: float = ALPHA):
        if self.df_resid <= 0 or self.params.size == 0:
            return np.empty((0, 2), dtype=float)
        t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=self.df_resid)
        lower = self.params - t_crit * self.bse
        upper = self.params + t_crit * self.bse
        return np.stack([lower, upper], axis=1)


class FederatedOLS:
    """Federated Ordinary Least Squares with statsmodels-like results."""

    def __init__(self, agg_client, *, fit_intercept: bool = True):
        self.agg_client = agg_client
        self.fit_intercept = fit_intercept
        self.results: FederatedOLSResults | None = None
        self.params = np.array([], dtype=float)
        self.bse = np.array([], dtype=float)
        self.tvalues = np.array([], dtype=float)
        self.pvalues = np.array([], dtype=float)
        self.df_resid = 0.0
        self.df_model = 0.0
        self.rse = float("nan")
        self.fvalue = float("nan")
        self.f_pvalue = float("nan")
        self.rsquared = 0.0
        self.rsquared_adj = 0.0
        self.nobs = 0
        self.rss = 0.0
        self.tss = 0.0
        self.sum_abs_resid = 0.0
        self.rank_ = 0
        self.xTx_inv_ = np.zeros((0, 0), dtype=float)
        self.cov_params = np.zeros((0, 0), dtype=float)

    def fit(self, X, y) -> FederatedOLSResults:
        """
        Unlike statsmodels’ OLS, we call fit with X/y, since the federated
        aggregate client will recompute sufficient statistics every time
        without storing the full federated dataset in the init of the class.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        stats_dict = self._collect_stats(X, y)

        coefficients = np.asarray(stats_dict["coefficients"], dtype=float)
        xTx_inv = np.asarray(stats_dict["xTx_inv"], dtype=float)
        rss = float(stats_dict["rss"])
        tss = float(stats_dict["tss"])
        sum_abs_resid = float(stats_dict["sum_abs_resid"])
        n_obs = int(stats_dict["n_obs"])

        rank = int(stats_dict.get("rank", 0))
        if self.fit_intercept:
            p = max(rank - 1, 0)
        else:
            p = rank

        summary = self._compute_summary(
            coefficients=coefficients,
            xTx_inv=xTx_inv,
            rss=rss,
            tss=tss,
            n_obs=n_obs,
            p=p,
        )

        params = np.asarray(summary["coefficients"], dtype=float)
        std_err = np.asarray(summary["std_err"], dtype=float)
        t_stats = np.asarray(summary["t_stats"], dtype=float)
        pvalues = np.asarray(summary["pvalues"], dtype=float)

        cov_params = xTx_inv * (summary["rse"] ** 2)

        results = FederatedOLSResults(
            params=params,
            bse=std_err,
            tvalues=t_stats,
            pvalues=pvalues,
            df_resid=summary["df_resid"],
            df_model=summary["df_model"],
            rse=summary["rse"],
            fvalue=summary["f_stat"],
            f_pvalue=summary["f_pvalue"],
            r_squared=summary["r_squared"],
            r_squared_adjusted=summary["r_squared_adjusted"],
            n_obs=summary["n_obs"],
            rss=rss,
            tss=tss,
            sum_abs_resid=sum_abs_resid,
            rank_=rank,
            xTx_inv_=xTx_inv,
            cov_params=cov_params,
        )

        self.results = results
        self.params = results.params
        self.bse = results.bse
        self.tvalues = results.tvalues
        self.pvalues = results.pvalues
        self.df_resid = results.df_resid
        self.df_model = results.df_model
        self.rse = results.rse
        self.fvalue = results.fvalue
        self.f_pvalue = results.f_pvalue
        self.rsquared = results.rsquared
        self.rsquared_adj = results.rsquared_adj
        self.nobs = results.nobs
        self.rss = results.rss
        self.tss = results.tss
        self.sum_abs_resid = results.sum_abs_resid
        self.rank_ = results.rank_
        self.xTx_inv_ = results.xTx_inv_
        self.cov_params = results.cov_params

        return results

    def conf_int(self, alpha: float = ALPHA):
        if self.results is None:
            return np.empty((0, 2), dtype=float)
        return self.results.conf_int(alpha)

    def _collect_stats(self, X: np.ndarray, y: np.ndarray):
        n_features = X.shape[1]

        xTx_local = X.T @ X if n_features > 0 else np.zeros((0, 0), dtype=float)
        xTy_local = X.T @ y if n_features > 0 else np.zeros((0, 1), dtype=float)
        n_obs_local = float(y.size)
        sum_y_local = float(y.sum())
        sum_sq_y_local = float((y**2).sum())

        if n_features > 0:
            xTx = np.asarray(self.agg_client.sum(xTx_local), dtype=float)
            xTy = np.asarray(self.agg_client.sum(xTy_local), dtype=float)
        else:
            xTx = np.zeros((0, 0), dtype=float)
            xTy = np.zeros((0, 1), dtype=float)

        n_obs = int(
            np.asarray(
                self.agg_client.sum(np.array([n_obs_local], dtype=float)),
                dtype=float,
            ).reshape(-1)[0]
        )
        sum_y = float(
            np.asarray(
                self.agg_client.sum(np.array([sum_y_local], dtype=float)),
                dtype=float,
            ).reshape(-1)[0]
        )
        sum_sq_y = float(
            np.asarray(
                self.agg_client.sum(np.array([sum_sq_y_local], dtype=float)),
                dtype=float,
            ).reshape(-1)[0]
        )

        if n_features > 0:
            xTx_inv = np.linalg.pinv(xTx)
            coefficients = xTx_inv @ xTy
            rank = int(np.linalg.matrix_rank(xTx))
        else:
            xTx_inv = np.zeros((0, 0), dtype=float)
            coefficients = np.zeros((0, 1), dtype=float)
            rank = 0

        if n_features > 0 and n_obs_local > 0:
            y_pred_local = X @ coefficients
            resid_local = y - y_pred_local
            rss_local = float(np.dot(resid_local.reshape(-1), resid_local.reshape(-1)))
            sum_abs_resid_local = float(np.abs(resid_local).sum())
        else:
            rss_local = 0.0
            sum_abs_resid_local = 0.0

        rss_arr = self.agg_client.sum(np.array([rss_local], dtype=float))
        sum_abs_resid_arr = self.agg_client.sum(
            np.array([sum_abs_resid_local], dtype=float)
        )
        rss = float(np.asarray(rss_arr, dtype=float).reshape(-1)[0])
        sum_abs_resid = float(np.asarray(sum_abs_resid_arr, dtype=float).reshape(-1)[0])

        if n_obs > 0:
            y_mean = sum_y / n_obs
            tss = sum_sq_y - 2.0 * y_mean * sum_y + n_obs * (y_mean**2)
        else:
            tss = 0.0

        return {
            "coefficients": coefficients.reshape(-1).tolist(),
            "xTx_inv": xTx_inv.tolist(),
            "rss": rss,
            "tss": tss,
            "sum_abs_resid": sum_abs_resid,
            "n_obs": n_obs,
            "rank": rank,
        }

    def _compute_summary(
        self,
        *,
        coefficients: np.ndarray,
        xTx_inv: np.ndarray,
        rss: float,
        tss: float,
        n_obs: int,
        p: int,
    ):
        df_resid = n_obs - p - 1
        df_model = p

        coeff_flat = coefficients.reshape(-1)
        if df_resid <= 0 or n_obs <= 0:
            nan_list = [float("nan")] * len(coeff_flat)
            return {
                "n_obs": int(n_obs),
                "df_resid": float(df_resid),
                "df_model": float(df_model),
                "rse": float("nan"),
                "r_squared": 0.0,
                "r_squared_adjusted": 0.0,
                "f_stat": float("nan"),
                "f_pvalue": float("nan"),
                "coefficients": coeff_flat.tolist(),
                "std_err": nan_list,
                "t_stats": nan_list,
                "pvalues": nan_list,
            }

        rse = float(np.sqrt(rss / df_resid))

        diag = np.diag(xTx_inv) if xTx_inv.size else np.array([], dtype=float)
        std_err = np.sqrt(np.maximum(diag, 0.0) * (rse**2))

        with np.errstate(divide="ignore", invalid="ignore"):
            t_stats = np.divide(
                coeff_flat,
                std_err,
                out=np.zeros_like(coeff_flat, dtype=float),
                where=std_err != 0,
            )

        pvalues = stats.t.sf(np.abs(t_stats), df=df_resid) * 2.0

        if tss > 0:
            r_squared = 1.0 - (rss / tss)
        else:
            r_squared = 0.0
        r_squared_adjusted = 1.0 - (1.0 - r_squared) * (n_obs - 1) / df_resid

        if rss == 0.0 or p == 0:
            f_stat = float("inf")
            f_pvalue = 0.0
        else:
            f_stat = float((tss - rss) * df_resid / (p * rss))
            f_pvalue = float(stats.f.sf(f_stat, dfn=p, dfd=df_resid))

        return {
            "n_obs": int(n_obs),
            "df_resid": float(df_resid),
            "df_model": float(df_model),
            "rse": rse,
            "r_squared": float(r_squared),
            "r_squared_adjusted": float(r_squared_adjusted),
            "f_stat": f_stat,
            "f_pvalue": f_pvalue,
            "coefficients": coeff_flat.tolist(),
            "std_err": std_err.tolist(),
            "t_stats": t_stats.tolist(),
            "pvalues": pvalues.tolist(),
        }
