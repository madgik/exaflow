"""FETA-owned logistic regression inference from completed global statistics."""

import numpy as np
from scipy import stats
from scipy.special import xlogy

from exaflow.worker_communication import BadUserInput

MAX_ITER = 50
TOL = 1e-4
ALPHA = 0.05


def _handle_logreg_errors(nobs: int, p: int, y_sum: float) -> None:
    if nobs <= p:
        msg = (
            "Logistic regression cannot run because the number of "
            "observations is smaller than the number of predictors. Please "
            "add more predictors or select more observations."
        )
        raise BadUserInput(msg)
    if min(y_sum, nobs - y_sum) <= p:
        msg = (
            "Logistic regression cannot run because the number of "
            "observations in one category is smaller than the number of "
            "predictors. Please add more predictors or select more "
            "observations for the category in question."
        )
        raise BadUserInput(msg)


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
