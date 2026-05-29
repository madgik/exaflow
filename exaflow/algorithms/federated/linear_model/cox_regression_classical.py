from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats

from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.utils.agg_client import AggregationClient
from exaflow.algorithms.federated.utils.interfaces import FederatedEstimator
from exaflow.algorithms.federated.utils.interfaces import FederatedEstimatorResults


def _as_1d_float(values) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def _safe_exp(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finfo = np.finfo(float)
    lower = np.log(finfo.tiny)
    upper = np.log(finfo.max)
    return np.exp(np.clip(values, lower, upper))


@dataclass(frozen=True)
class FederatedClassicalCoxRegressionResults(FederatedEstimatorResults):
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
    n_covariates: int
    n_unique_event_times: int
    df_model: int
    df_resid: int
    ll: float
    indep_vars: list[str]
    ties: str
    n_iter: int
    converged: bool
    score_norm: float
    step_norm: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return _safe_exp(X @ self.params.reshape(-1, 1)).reshape(-1)

    def summary(self) -> dict:
        return {
            "n_obs": int(self.nobs),
            "n_events": int(self.n_events),
            "n_covariates": int(self.n_covariates),
            "n_unique_event_times": int(self.n_unique_event_times),
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
            "n_iter": int(self.n_iter),
            "converged": bool(self.converged),
            "score_norm": float(self.score_norm),
            "step_norm": float(self.step_norm),
            "method": "classical_cox_partial_likelihood",
        }


class FederatedClassicalCoxRegression(FederatedEstimator):
    def __init__(
        self,
        *,
        ties: str = "breslow",
        tol: float = 1e-6,
        max_iter: int = 100,
        ridge: float = 1e-8,
        max_step_norm: float = 5.0,
        verbose: bool = False,
    ) -> None:
        if ties != "breslow":
            raise BadInputError(
                "FederatedClassicalCoxRegression currently supports ties='breslow' only."
            )
        if tol <= 0:
            raise BadInputError("tol must be positive.")
        if max_iter <= 0:
            raise BadInputError("max_iter must be positive.")
        if ridge < 0:
            raise BadInputError("ridge must be non-negative.")
        if max_step_norm <= 0:
            raise BadInputError("max_step_norm must be positive.")

        self.ties = ties
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.ridge = float(ridge)
        self.max_step_norm = float(max_step_norm)
        self.verbose = bool(verbose)
        self.results: FederatedClassicalCoxRegressionResults | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        agg_client: AggregationClient,
        feature_names: Sequence[str] | None = None,
    ) -> FederatedClassicalCoxRegressionResults:
        X = np.asarray(X, dtype=float)
        times, events = self._parse_survival_target(y)
        if X.shape[0] != times.shape[0]:
            raise BadInputError(
                "Design matrix row count must match the number of survival target rows."
            )
        X, times, events = self._validate_inputs(X, times, events)
        X, times, events = self._sort_by_time(X, times, events)
        event_times = self._get_event_time_grid(times, events, agg_client=agg_client)
        event_index = self._build_event_index(event_times)
        n_obs_local = int(X.shape[0])
        n_events_local = int(events.sum())
        n_covariates = int(X.shape[1])
        indep_vars = list(feature_names or [f"x{i}" for i in range(n_covariates)])

        d_local, e1_local = self._compute_local_event_blocks(
            X,
            times,
            events,
            event_times,
            event_index,
        )
        d_global, e1_global, n_obs, n_events = self._aggregate_static_blocks(
            d_local=d_local,
            e1_local=e1_local,
            n_obs_local=n_obs_local,
            n_events_local=n_events_local,
            agg_client=agg_client,
        )
        if n_events <= 0:
            raise BadInputError(
                "cox_regression_classical requires at least one observed event globally."
            )

        beta = np.zeros(n_covariates, dtype=float)
        converged = False
        step_norm = 0.0
        score_norm = float("inf")
        n_iter_done = 0

        for iteration in range(1, self.max_iter + 1):
            s0_local, s1_local, s2_local = self._compute_local_risk_blocks(
                X,
                times,
                event_times,
                beta,
            )
            s0_global, s1_global, s2_global = self._aggregate_risk_blocks(
                s0_local=s0_local,
                s1_local=s1_local,
                s2_local=s2_local,
                agg_client=agg_client,
            )
            loglik, score, info = self._compute_breslow_loglik_score_info(
                beta=beta,
                d=d_global,
                e1=e1_global,
                s0=s0_global,
                s1=s1_global,
                s2=s2_global,
            )
            beta_new, step_norm = self._newton_step(beta, score, info)
            converged, _, score_norm = self._has_converged(beta, beta_new, score)
            beta = beta_new
            n_iter_done = iteration

            if self.verbose:
                print(
                    "[cox_regression_classical]",
                    f"iter={iteration}",
                    f"loglik={loglik:.6f}",
                    f"score_norm={score_norm:.6e}",
                    f"step_norm={step_norm:.6e}",
                )
            if converged:
                break

        s0_local, s1_local, s2_local = self._compute_local_risk_blocks(
            X,
            times,
            event_times,
            beta,
        )
        s0_global, s1_global, s2_global = self._aggregate_risk_blocks(
            s0_local=s0_local,
            s1_local=s1_local,
            s2_local=s2_local,
            agg_client=agg_client,
        )
        loglik, score, info = self._compute_breslow_loglik_score_info(
            beta=beta,
            d=d_global,
            e1=e1_global,
            s0=s0_global,
            s1=s1_global,
            s2=s2_global,
        )
        score_norm = float(np.linalg.norm(score))
        std_err, z_scores, pvalues, lower_ci, upper_ci = self._compute_inference(
            beta,
            info,
        )

        results = FederatedClassicalCoxRegressionResults(
            params=beta,
            hazard_ratios=_safe_exp(beta),
            std_err=std_err,
            lower_ci=lower_ci,
            upper_ci=upper_ci,
            hr_lower_ci=_safe_exp(lower_ci),
            hr_upper_ci=_safe_exp(upper_ci),
            z_scores=z_scores,
            pvalues=pvalues,
            nobs=int(n_obs),
            n_events=int(n_events),
            n_covariates=n_covariates,
            n_unique_event_times=int(len(event_times)),
            df_model=n_covariates,
            df_resid=int(n_obs - n_covariates),
            ll=float(loglik),
            indep_vars=indep_vars,
            ties=self.ties,
            n_iter=n_iter_done,
            converged=converged,
            score_norm=score_norm,
            step_norm=float(step_norm),
        )
        self.results = results
        return results

    @staticmethod
    def _parse_survival_target(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y = np.asarray(y, dtype=float)

        if y.ndim != 2:
            raise BadInputError(
                "Survival target y must be a 2-dimensional array with columns [time, event]."
            )
        if y.shape[1] != 2:
            raise BadInputError(
                "Survival target y must have shape (n_obs, 2) with columns [time, event]."
            )
        if y.shape[0] == 0:
            raise BadInputError(
                "Survival target y must contain at least one observation."
            )
        if not np.isfinite(y).all():
            raise BadInputError("Survival target y contains non-finite values.")

        times = np.asarray(y[:, 0], dtype=float).reshape(-1)
        events = np.asarray(y[:, 1], dtype=float).reshape(-1)

        unique_events = np.unique(events)
        if not np.isin(unique_events, [0.0, 1.0]).all():
            raise BadInputError(
                "Survival target event column must be binary with values in {0, 1}."
            )

        return times, events

    @staticmethod
    def _validate_inputs(
        X: np.ndarray,
        times: np.ndarray,
        events: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        times = _as_1d_float(times)
        events = _as_1d_float(events)
        if X.ndim != 2:
            raise BadInputError(f"X must be 2D, got shape {X.shape}.")
        if times.shape != (X.shape[0],):
            raise BadInputError("times must have the same length as X.")
        if events.shape != (X.shape[0],):
            raise BadInputError("events must have the same length as X.")
        if X.shape[0] == 0:
            raise BadInputError(
                "cox_regression_classical requires at least one observation."
            )
        if X.shape[1] == 0:
            raise BadInputError(
                "cox_regression_classical requires at least one covariate."
            )
        if not np.isfinite(X).all():
            raise BadInputError("X contains NaN/Inf.")
        if not np.isfinite(times).all():
            raise BadInputError("times contains NaN/Inf.")
        if not np.isfinite(events).all():
            raise BadInputError("events contains NaN/Inf.")
        if np.min(times) <= 0:
            raise BadInputError("All time values must be strictly positive.")
        unique_events = np.unique(events)
        if not np.isin(unique_events, [0.0, 1.0]).all():
            raise BadInputError(
                "cox_regression_classical expects a binary event indicator."
            )
        return X, times, events

    @staticmethod
    def _sort_by_time(
        X: np.ndarray,
        times: np.ndarray,
        events: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        order = np.argsort(times, kind="mergesort")
        return X[order], times[order], events[order]

    @staticmethod
    def _get_event_time_grid(
        times: np.ndarray,
        events: np.ndarray,
        *,
        agg_client: AggregationClient,
    ) -> np.ndarray:
        local_event_times = np.unique(times[events > 0]).astype(float)
        global_event_times = np.asarray(
            agg_client.union(local_event_times.tolist()),
            dtype=float,
        )
        if global_event_times.size == 0:
            raise BadInputError(
                "cox_regression_classical requires at least one observed event."
            )
        return np.sort(global_event_times)

    @staticmethod
    def _build_event_index(event_times: np.ndarray) -> dict[float, int]:
        return {float(event_time): idx for idx, event_time in enumerate(event_times)}

    @staticmethod
    def _compute_local_event_blocks(
        X: np.ndarray,
        times: np.ndarray,
        events: np.ndarray,
        event_times: np.ndarray,
        event_index: dict[float, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        n_event_times = len(event_times)
        n_covariates = X.shape[1]
        d_local = np.zeros(n_event_times, dtype=float)
        e1_local = np.zeros((n_event_times, n_covariates), dtype=float)

        event_rows = np.where(events > 0)[0]
        for row_idx in event_rows:
            global_idx = event_index[float(times[row_idx])]
            d_local[global_idx] += 1.0
            e1_local[global_idx] += X[row_idx]
        return d_local, e1_local

    def _aggregate_static_blocks(
        self,
        *,
        d_local: np.ndarray,
        e1_local: np.ndarray,
        n_obs_local: int,
        n_events_local: int,
        agg_client: AggregationClient,
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        fused = np.concatenate(
            [
                d_local.reshape(-1),
                e1_local.reshape(-1),
                np.array([float(n_obs_local), float(n_events_local)], dtype=float),
            ]
        )
        fused_sum = np.asarray(agg_client.sum(fused), dtype=float)
        n_event_times = d_local.shape[0]
        n_covariates = e1_local.shape[1]
        d_global = fused_sum[:n_event_times]
        e1_start = n_event_times
        e1_end = e1_start + n_event_times * n_covariates
        e1_global = fused_sum[e1_start:e1_end].reshape(n_event_times, n_covariates)
        n_obs = int(round(float(fused_sum[e1_end])))
        n_events = int(round(float(fused_sum[e1_end + 1])))
        return d_global, e1_global, n_obs, n_events

    @staticmethod
    def _compute_local_risk_blocks(
        X: np.ndarray,
        times: np.ndarray,
        event_times: np.ndarray,
        beta: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        eta = X @ beta
        weights = _safe_exp(eta)
        weighted_X = X * weights[:, None]
        weighted_outer = X[:, :, None] * weighted_X[:, None, :]

        n_obs = X.shape[0]
        n_covariates = X.shape[1]
        s0_suffix = np.zeros(n_obs + 1, dtype=float)
        s1_suffix = np.zeros((n_obs + 1, n_covariates), dtype=float)
        s2_suffix = np.zeros((n_obs + 1, n_covariates, n_covariates), dtype=float)

        s0_suffix[:-1] = np.cumsum(weights[::-1], axis=0)[::-1]
        s1_suffix[:-1] = np.cumsum(weighted_X[::-1], axis=0)[::-1]
        s2_suffix[:-1] = np.cumsum(weighted_outer[::-1], axis=0)[::-1]

        indices = np.searchsorted(times, event_times, side="left")
        s0_local = s0_suffix[indices]
        s1_local = s1_suffix[indices]
        s2_local = s2_suffix[indices]
        return s0_local, s1_local, s2_local

    def _aggregate_risk_blocks(
        self,
        *,
        s0_local: np.ndarray,
        s1_local: np.ndarray,
        s2_local: np.ndarray,
        agg_client: AggregationClient,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        fused = np.concatenate(
            [
                s0_local.reshape(-1),
                s1_local.reshape(-1),
                s2_local.reshape(-1),
            ]
        )
        fused_sum = np.asarray(agg_client.sum(fused), dtype=float)
        n_event_times = s0_local.shape[0]
        n_covariates = s1_local.shape[1]
        s0_end = n_event_times
        s1_end = s0_end + n_event_times * n_covariates
        s0_global = fused_sum[:s0_end]
        s1_global = fused_sum[s0_end:s1_end].reshape(n_event_times, n_covariates)
        s2_global = fused_sum[s1_end:].reshape(
            n_event_times,
            n_covariates,
            n_covariates,
        )
        return s0_global, s1_global, s2_global

    @staticmethod
    def _compute_breslow_loglik_score_info(
        *,
        beta: np.ndarray,
        d: np.ndarray,
        e1: np.ndarray,
        s0: np.ndarray,
        s1: np.ndarray,
        s2: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        if np.any(s0 <= 0):
            raise BadInputError(
                "Encountered an empty or invalid risk set while fitting cox_regression_classical."
            )

        linear_event_term = float(np.sum(e1 @ beta))
        loglik = linear_event_term - float(np.sum(d * np.log(s0)))

        mean_x = s1 / s0[:, None]
        score = np.sum(e1 - d[:, None] * mean_x, axis=0)

        outer_means = mean_x[:, :, None] * mean_x[:, None, :]
        info_terms = (s2 / s0[:, None, None]) - outer_means
        info = np.sum(d[:, None, None] * info_terms, axis=0)
        info = 0.5 * (info + info.T)
        return loglik, score, info

    def _newton_step(
        self,
        beta: np.ndarray,
        score: np.ndarray,
        info: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        eye = np.eye(info.shape[0], dtype=float)
        lam = max(self.ridge, 0.0)
        delta = None

        for _ in range(6):
            try:
                delta = np.linalg.solve(info + lam * eye, score)
                break
            except np.linalg.LinAlgError:
                lam = max(lam * 10.0, 1e-12)

        if delta is None:
            delta = np.linalg.pinv(info + max(lam, 1e-6) * eye) @ score

        step_norm = float(np.linalg.norm(delta))
        if step_norm > self.max_step_norm and step_norm > 0.0:
            delta *= self.max_step_norm / step_norm
            step_norm = float(np.linalg.norm(delta))
        return beta + delta, step_norm

    def _has_converged(
        self,
        beta_old: np.ndarray,
        beta_new: np.ndarray,
        score: np.ndarray,
    ) -> tuple[bool, float, float]:
        max_delta = float(np.max(np.abs(beta_new - beta_old)))
        score_norm = float(np.linalg.norm(score, ord=np.inf))
        converged = max_delta < self.tol and score_norm < np.sqrt(self.tol)
        return converged, max_delta, score_norm

    @staticmethod
    def _compute_inference(
        beta: np.ndarray,
        info: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        eye = np.eye(info.shape[0], dtype=float)
        try:
            covariance = np.linalg.inv(info + 1e-10 * eye)
        except np.linalg.LinAlgError:
            covariance = np.linalg.pinv(info + 1e-8 * eye)

        variances = np.clip(np.diag(covariance), 0.0, None)
        std_err = np.sqrt(variances)
        z_scores = np.divide(
            beta,
            std_err,
            out=np.zeros_like(beta, dtype=float),
            where=std_err > 0,
        )
        pvalues = stats.norm.sf(np.abs(z_scores)) * 2.0
        z_975 = stats.norm.ppf(0.975)
        lower_ci = beta - z_975 * std_err
        upper_ci = beta + z_975 * std_err
        return std_err, z_scores, pvalues, lower_ci, upper_ci
