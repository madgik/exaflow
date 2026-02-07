from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.special import expit
from scipy.special import xlogy

from exaflow.algorithms.federated.agg_client import AggregationClient
from exaflow.algorithms.federated.interfaces import FederatedEstimator
from exaflow.algorithms.federated.interfaces import FederatedEstimatorResults
from exaflow.algorithms.federated.utils import BadInputError

MAX_ITER = 50
TOL = 1e-4
ALPHA = 0.05


class FederatedLogisticRegressionResults(FederatedEstimatorResults):
    """Container for fitted federated logistic regression statistics."""

    nobs: int

    def __init__(
        self,
        *,
        params,
        hessian_inverse,
        ll,
        n_obs,
        y_sum,
        stderr,
        lower_ci,
        upper_ci,
        z_scores,
        pvalues,
        df_model,
        df_resid,
        r_squared_cs,
        r_squared_mcf,
        ll0,
        aic,
        bic,
        fit_intercept,
    ):
        self.params = np.asarray(params, dtype=float)
        self.coefficients = self.params
        self.hessian_inverse = np.asarray(hessian_inverse, dtype=float)
        self.ll = float(ll)
        self.nobs = int(n_obs)
        self.y_sum = float(y_sum)
        self.stderr = np.asarray(stderr, dtype=float)
        self.lower_ci = np.asarray(lower_ci, dtype=float)
        self.upper_ci = np.asarray(upper_ci, dtype=float)
        self.z_scores = np.asarray(z_scores, dtype=float)
        self.pvalues = np.asarray(pvalues, dtype=float)
        self.df_model = int(df_model)
        self.df_resid = int(df_resid)
        self.r_squared_cs = float(r_squared_cs)
        self.r_squared_mcf = float(r_squared_mcf)
        self.ll0 = float(ll0)
        self.aic = float(aic)
        self.bic = float(bic)
        self.fit_intercept = bool(fit_intercept)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.params.size == 0:
            return np.zeros((X.shape[0],), dtype=float)
        X = np.asarray(X, dtype=float)
        if self.fit_intercept:
            X = FederatedLogisticRegression._add_intercept(X)
        logits = X @ self.params.reshape(-1, 1)
        return expit(logits).reshape(-1)

    def summary(self) -> dict:
        return {
            "n_obs": int(self.nobs),
            "coefficients": self.params.tolist(),
            "stderr": self.stderr.tolist(),
            "lower_ci": self.lower_ci.tolist(),
            "upper_ci": self.upper_ci.tolist(),
            "z_scores": self.z_scores.tolist(),
            "pvalues": self.pvalues.tolist(),
            "df_model": int(self.df_model),
            "df_resid": int(self.df_resid),
            "r_squared_cs": float(self.r_squared_cs),
            "r_squared_mcf": float(self.r_squared_mcf),
            "ll0": float(self.ll0),
            "ll": float(self.ll),
            "aic": float(self.aic),
            "bic": float(self.bic),
        }


class FederatedLogisticRegression(FederatedEstimator):
    """Federated logistic regression with statsmodels-like results."""

    def __init__(self, *, fit_intercept: bool = True) -> None:
        self.fit_intercept = fit_intercept
        self.results: FederatedLogisticRegressionResults | None = None
        self.params = np.array([], dtype=float)
        self.hessian_inverse = np.zeros((0, 0), dtype=float)
        self.ll = float("nan")
        self.nobs = 0
        self.y_sum = 0.0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        agg_client: AggregationClient,
    ) -> FederatedLogisticRegressionResults:
        X = np.asarray(X, dtype=float)
        if self.fit_intercept:
            X = self._add_intercept(X)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        stats_dict = self._collect_stats(X, y, agg_client)
        coefficients = np.asarray(stats_dict["coefficients"], dtype=float)
        hessian_inverse = np.asarray(stats_dict["hessian_inverse"], dtype=float)
        ll = float(stats_dict["ll"])
        n_obs = int(stats_dict["n_obs"])
        y_sum = float(stats_dict["y_sum"])

        summary = self._compute_summary(
            coefficients=coefficients.reshape(-1),
            h_inv=hessian_inverse,
            ll=ll,
            n_obs=n_obs,
            y_sum=y_sum,
            alpha=ALPHA,
        )

        results = FederatedLogisticRegressionResults(
            params=summary["coefficients"],
            hessian_inverse=hessian_inverse,
            ll=summary["ll"],
            n_obs=summary["n_obs"],
            y_sum=y_sum,
            stderr=summary["stderr"],
            lower_ci=summary["lower_ci"],
            upper_ci=summary["upper_ci"],
            z_scores=summary["z_scores"],
            pvalues=summary["pvalues"],
            df_model=summary["df_model"],
            df_resid=summary["df_resid"],
            r_squared_cs=summary["r_squared_cs"],
            r_squared_mcf=summary["r_squared_mcf"],
            ll0=summary["ll0"],
            aic=summary["aic"],
            bic=summary["bic"],
            fit_intercept=self.fit_intercept,
        )

        self.results = results
        self.params = results.params
        self.hessian_inverse = results.hessian_inverse
        self.ll = results.ll
        self.nobs = results.nobs
        self.y_sum = results.y_sum

        return results

    @staticmethod
    def coerce_positive_class(series, positive_class):
        """
        Try to cast `positive_class` to the dtype of the provided series.
        This avoids mismatches such as comparing numeric columns to string labels.
        """
        if positive_class is None:
            return positive_class

        try:
            dtype = series.dtype
        except Exception:
            return positive_class

        try:
            if hasattr(dtype, "type"):
                return dtype.type(positive_class)
        except Exception:
            pass

        try:
            first_valid = series.dropna()
            if len(first_valid):
                sample = first_valid.iloc[0]
                return type(sample)(positive_class)
        except Exception:
            pass

        return positive_class

    @staticmethod
    def _add_intercept(X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_rows = X.shape[0]
        intercept = np.ones((n_rows, 1), dtype=float)
        return np.hstack([intercept, X])

    @staticmethod
    def _handle_logreg_errors(nobs: int, p: int, y_sum: float) -> None:
        if nobs <= p:
            msg = (
                "Logistic regression cannot run because the number of "
                "observations is smaller than the number of predictors. Please "
                "add more predictors or select more observations."
            )
            raise BadInputError(msg)
        if min(y_sum, nobs - y_sum) <= p:
            msg = (
                "Logistic regression cannot run because the number of "
                "observations in one category is smaller than the number of "
                "predictors. Please add more predictors or select more "
                "observations for the category in question."
            )
            raise BadInputError(msg)

    @staticmethod
    def _max_abs(values: np.ndarray) -> float:
        return float(np.max(np.abs(values))) if len(values) else 0.0

    def _collect_stats(
        self, X: np.ndarray, y: np.ndarray, agg_client: AggregationClient
    ) -> dict:
        if X.ndim != 2:
            X = np.atleast_2d(X)
        if y.ndim == 2 and y.shape[1] != 1:
            y = y.reshape(-1, 1)
        if X.shape[0] != y.shape[0]:
            if X.shape[1] == y.shape[0]:
                X = X.T
            else:
                raise BadInputError(
                    "Design matrix row count does not match target size for logistic regression."
                )

        n_obs_local = int(y.size)
        y_sum_local = float(y.sum())

        total_n_obs_arr = agg_client.sum(np.array([float(n_obs_local)], dtype=float))
        total_y_sum_arr = agg_client.sum(np.array([float(y_sum_local)], dtype=float))
        total_n_obs = int(np.asarray(total_n_obs_arr, dtype=float).reshape(-1)[0])
        total_y_sum = float(np.asarray(total_y_sum_arr, dtype=float).reshape(-1)[0])

        n_features = X.shape[1]
        self._handle_logreg_errors(total_n_obs, n_features, total_y_sum)

        coeff = np.zeros((n_features, 1), dtype=float)
        h_inv = np.eye(n_features, dtype=float)
        ll = 0.0

        for _ in range(MAX_ITER):
            eta = X @ coeff
            mu = expit(eta)
            w = mu * (1.0 - mu)

            grad_local = np.einsum("ji,j->i", X, (y - mu).reshape(-1))
            h_local = np.einsum("ji,j,jk->ik", X, w.reshape(-1), X)
            ll_local = np.sum(xlogy(y, mu) + xlogy(1 - y, 1 - mu))

            grad_arr = agg_client.sum(grad_local)
            h_arr = agg_client.sum(h_local)
            ll_arr = agg_client.sum(np.array([float(ll_local)], dtype=float))

            grad = np.asarray(grad_arr, dtype=float)
            hessian = np.asarray(h_arr, dtype=float)
            ll = float(np.asarray(ll_arr, dtype=float).reshape(-1)[0])

            try:
                h_inv = np.linalg.inv(hessian)
            except np.linalg.LinAlgError:
                h_inv = np.linalg.pinv(hessian)

            coeff = coeff + h_inv @ grad.reshape(-1, 1)

            if self._max_abs(grad) <= TOL:
                break
        else:
            raise BadInputError("Logistic regression cannot converge. Cancelling run.")

        return {
            "coefficients": coeff.reshape(-1).tolist(),
            "hessian_inverse": h_inv.tolist(),
            "ll": ll,
            "n_obs": total_n_obs,
            "y_sum": total_y_sum,
        }

    @staticmethod
    def _compute_summary(
        *,
        coefficients: np.ndarray,
        h_inv: np.ndarray,
        ll: float,
        n_obs: int,
        y_sum: float,
        alpha: float,
    ) -> dict:
        stderr = np.sqrt(np.diag(h_inv)) if h_inv.size else np.array([], dtype=float)
        z_scores = np.divide(
            coefficients,
            stderr,
            out=np.zeros_like(coefficients, dtype=float),
            where=stderr != 0,
        )
        pvalues = stats.norm.sf(np.abs(z_scores)) * 2.0

        z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
        lower_ci = coefficients - z_crit * stderr
        upper_ci = coefficients + z_crit * stderr

        df_model = len(coefficients) - 1
        df_resid = n_obs - len(coefficients)

        y_mean = y_sum / n_obs if n_obs else 0.0
        ll0 = float(xlogy(y_sum, y_mean) + xlogy(n_obs - y_sum, 1.0 - y_mean))

        aic = 2 * len(coefficients) - 2 * ll
        bic = np.log(n_obs) * len(coefficients) - 2 * ll if n_obs else float("inf")

        if np.isclose(ll, 0.0) and np.isclose(ll0, 0.0):
            r2_mcf = 1.0
        else:
            r2_mcf = 1.0 - (ll / ll0)
        r2_cs = 1.0 - np.exp(2.0 * (ll0 - ll) / n_obs) if n_obs else 0.0

        return {
            "n_obs": int(n_obs),
            "coefficients": coefficients.tolist(),
            "stderr": stderr.tolist(),
            "lower_ci": lower_ci.tolist(),
            "upper_ci": upper_ci.tolist(),
            "z_scores": z_scores.tolist(),
            "pvalues": pvalues.tolist(),
            "df_model": int(df_model),
            "df_resid": int(df_resid),
            "r_squared_cs": float(r2_cs),
            "r_squared_mcf": float(r2_mcf),
            "ll0": ll0,
            "ll": float(ll),
            "aic": aic,
            "bic": bic,
        }
