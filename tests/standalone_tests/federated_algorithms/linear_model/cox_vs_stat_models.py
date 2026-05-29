from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats
from statsmodels.duration.hazard_regression import PHReg

from exaflow.algorithms.federated.utils import BadInputError


@dataclass(frozen=True)
class CoxReferenceResults:
    params: np.ndarray
    hazard_ratios: np.ndarray
    std_err: np.ndarray
    lower_ci: np.ndarray
    upper_ci: np.ndarray
    hr_lower_ci: np.ndarray
    hr_upper_ci: np.ndarray
    z_scores: np.ndarray
    pvalues: np.ndarray
    nobs: int
    n_events: int
    df_model: int
    df_resid: int
    ll: float
    indep_vars: list[str]
    ties: str

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.exp(X @ self.params.reshape(-1, 1)).reshape(-1)

    def summary(self) -> dict:
        return {
            "n_obs": int(self.nobs),
            "n_events": int(self.n_events),
            "coefficients": self.params.tolist(),
            "hazard_ratios": self.hazard_ratios.tolist(),
            "std_err": self.std_err.tolist(),
            "lower_ci": self.lower_ci.tolist(),
            "upper_ci": self.upper_ci.tolist(),
            "hr_lower_ci": self.hr_lower_ci.tolist(),
            "hr_upper_ci": self.hr_upper_ci.tolist(),
            "z_scores": self.z_scores.tolist(),
            "pvalues": self.pvalues.tolist(),
            "df_model": int(self.df_model),
            "df_resid": int(self.df_resid),
            "ll": float(self.ll),
            "indep_vars": list(self.indep_vars),
            "ties": self.ties,
            "method": "cox_reference",
        }


class CoxReferencePH:
    """Centralized classical Cox PH reference fit using statsmodels PHReg."""

    def __init__(self, *, ties: str = "breslow") -> None:
        if ties != "breslow":
            raise BadInputError(
                "CoxReferencePH currently supports ties='breslow' only."
            )
        self.ties = ties
        self.results: CoxReferenceResults | None = None

    def fit(
        self,
        X: np.ndarray,
        times: np.ndarray,
        events: np.ndarray,
        *,
        feature_names: Sequence[str] | None = None,
    ) -> CoxReferenceResults:
        X = np.asarray(X, dtype=float)
        times = np.asarray(times, dtype=float).reshape(-1)
        events = np.asarray(events, dtype=float).reshape(-1)
        self._validate_inputs(X, times, events)

        model = PHReg(endog=times, exog=X, status=events, ties=self.ties)
        fit = model.fit(disp=0)

        params = np.asarray(fit.params, dtype=float)
        bse = np.asarray(fit.bse, dtype=float)
        z_scores = np.divide(
            params,
            bse,
            out=np.zeros_like(params, dtype=float),
            where=bse != 0,
        )
        pvalues = stats.norm.sf(np.abs(z_scores)) * 2.0
        conf_int = np.asarray(fit.conf_int(), dtype=float)
        lower_ci = conf_int[:, 0]
        upper_ci = conf_int[:, 1]
        hazard_ratios = np.exp(params)
        hr_lower_ci = np.exp(lower_ci)
        hr_upper_ci = np.exp(upper_ci)
        names = list(feature_names or [f"x{i}" for i in range(X.shape[1])])

        results = CoxReferenceResults(
            params=params,
            hazard_ratios=hazard_ratios,
            std_err=bse,
            lower_ci=lower_ci,
            upper_ci=upper_ci,
            hr_lower_ci=hr_lower_ci,
            hr_upper_ci=hr_upper_ci,
            z_scores=z_scores,
            pvalues=pvalues,
            nobs=int(len(times)),
            n_events=int(events.sum()),
            df_model=int(len(params)),
            df_resid=int(len(times) - len(params)),
            ll=float(fit.llf),
            indep_vars=names,
            ties=self.ties,
        )
        self.results = results
        return results

    @staticmethod
    def _validate_inputs(X: np.ndarray, times: np.ndarray, events: np.ndarray) -> None:
        if X.ndim != 2:
            raise BadInputError(f"X must be 2D, got shape {X.shape}.")
        if times.shape != (X.shape[0],):
            raise BadInputError("times must have the same length as X.")
        if events.shape != (X.shape[0],):
            raise BadInputError("events must have the same length as X.")
        if X.shape[0] == 0:
            raise BadInputError("CoxReferencePH requires at least one observation.")
        if np.min(times) <= 0:
            raise BadInputError("All time values must be strictly positive.")
        if not np.isin(np.unique(events), [0.0, 1.0]).all():
            raise BadInputError("events must be binary.")
        if int(events.sum()) <= 0:
            raise BadInputError("CoxReferencePH requires at least one observed event.")
