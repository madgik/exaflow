from __future__ import annotations

import numpy as np
from statsmodels.duration.hazard_regression import PHReg


def fit_statsmodels_cox(X: np.ndarray, times: np.ndarray, events: np.ndarray) -> dict:
    """Fit the centralized Breslow reference and extract the values under comparison."""
    model = PHReg(endog=times, exog=X, status=events, ties="breslow")
    fit = model.fit(disp=0)

    return {
        "n_obs": model.nobs,
        "n_events": int(model.status.sum()),
        "coefficients": fit.params,
        "hazard_ratios": np.exp(fit.params),
        "std_err": fit.bse,
        "pvalues": fit.pvalues,
        "df_model": int(fit.df_model),
        "df_resid": int(fit.df_resid),
        "ll": float(fit.llf),
        "indep_vars": [f"x{i}" for i in range(X.shape[1])],
        "ties": model.ties,
    }
