from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from exaflow.algorithms.federated.linear_model.logistic_regression import (
    FederatedLogisticRegression,
)
from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.utils.agg_client import AggregationClient
from exaflow.algorithms.federated.utils.interfaces import FederatedEstimator
from exaflow.algorithms.federated.utils.interfaces import FederatedEstimatorResults

ALLOWED_TIME_GRID_STRATEGIES = {"distinct_event_times", "uniform"}


def _as_1d_float(values) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def _strictly_positive_upper_edge(max_time: float) -> float:
    upper = float(np.nextafter(max_time, np.inf))
    if upper <= max_time:
        upper = max_time + 1.0
    return upper


def _safe_exp(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finfo = np.finfo(float)
    lower = np.log(finfo.tiny)
    upper = np.log(finfo.max)
    return np.exp(np.clip(values, lower, upper))


def _validate_binary_events(events: np.ndarray) -> None:
    if events.ndim != 1:
        raise BadInputError("events must be one-dimensional.")
    unique_values = np.unique(events)
    if not np.isin(unique_values, [0.0, 1.0]).all():
        raise BadInputError("cox_regression_stacked expects a binary event indicator.")


def _grouped_last_indices(size: int, n_groups: int) -> np.ndarray:
    if n_groups <= 0:
        raise BadInputError("n_groups must be positive.")
    if size <= n_groups:
        return np.arange(size, dtype=int)
    groups = np.array_split(np.arange(size, dtype=int), n_groups)
    return np.asarray([group[-1] for group in groups if len(group)], dtype=int)


@dataclass(frozen=True)
class FederatedStackedCoxRegressionResults(FederatedEstimatorResults):
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
    n_stacked_rows: int
    n_covariates: int
    df_model: int
    df_resid: int
    r_squared_cs: float
    r_squared_mcf: float
    ll0: float
    ll: float
    aic: float
    bic: float
    indep_vars: list[str]
    time_grid_strategy: str
    n_time_bins_used: int
    time_bins: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return _safe_exp(X @ self.params.reshape(-1, 1)).reshape(-1)

    def summary(self) -> dict:
        return {
            "n_obs": int(self.nobs),
            "n_events": int(self.n_events),
            "n_stacked_rows": int(self.n_stacked_rows),
            "n_covariates": int(self.n_covariates),
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
            "r_squared_cs": float(self.r_squared_cs),
            "r_squared_mcf": float(self.r_squared_mcf),
            "ll0": float(self.ll0),
            "ll": float(self.ll),
            "aic": float(self.aic),
            "bic": float(self.bic),
            "indep_vars": list(self.indep_vars),
            "time_grid_strategy": self.time_grid_strategy,
            "n_time_bins_used": int(self.n_time_bins_used),
            "method": "survival_stacking",
        }


class FederatedStackedCoxRegression(FederatedEstimator):
    """
    Cox-like survival model fitted via survival stacking and federated logistic
    regression.

    The fitted covariate coefficients approximate Cox proportional hazards
    log-hazard ratios, while the time-bin dummy coefficients are treated as
    nuisance baseline-hazard terms and are not exposed in the public summary.
    """

    def __init__(
        self,
        *,
        time_grid_strategy: str = "distinct_event_times",
        n_time_bins: int = 10,
    ) -> None:
        if time_grid_strategy not in ALLOWED_TIME_GRID_STRATEGIES:
            raise BadInputError(
                "Unsupported time_grid_strategy. Expected one of "
                f"{sorted(ALLOWED_TIME_GRID_STRATEGIES)}."
            )
        if n_time_bins <= 0:
            raise BadInputError("n_time_bins must be positive.")
        self.time_grid_strategy = time_grid_strategy
        self.n_time_bins = int(n_time_bins)
        self.results: FederatedStackedCoxRegressionResults | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        agg_client: AggregationClient,
        feature_names: Sequence[str] | None = None,
    ) -> FederatedStackedCoxRegressionResults:
        X = np.asarray(X, dtype=float)
        times, events = self._parse_survival_target(y)
        if X.shape[0] != times.shape[0]:
            raise BadInputError(
                "Design matrix row count must match the number of survival target rows."
            )
        times = _as_1d_float(times)
        events = _as_1d_float(events)
        self._validate_fit_inputs(X, times, events)
        global_n_events = int(
            np.asarray(
                agg_client.sum(np.array([float(events.sum())], dtype=float)),
                dtype=float,
            ).reshape(-1)[0]
        )
        if global_n_events <= 0:
            raise BadInputError(
                "cox_regression_stacked requires at least one observed event globally."
            )
        max_time_bins = self._recommended_max_time_bins(
            n_events=global_n_events,
            n_covariates=X.shape[1],
        )

        time_bins = self.compute_time_bins(
            times=times,
            events=events,
            agg_client=agg_client,
            max_time_bins=max_time_bins,
        )
        stacked_X, stacked_y = self.stack(
            X=X,
            times=times,
            events=events,
            time_bins=time_bins,
        )
        global_nobs = int(
            np.asarray(
                agg_client.sum(np.array([float(len(times))], dtype=float)),
                dtype=float,
            ).reshape(-1)[0]
        )
        global_n_stacked_rows = int(
            np.asarray(
                agg_client.sum(np.array([float(len(stacked_y))], dtype=float)),
                dtype=float,
            ).reshape(-1)[0]
        )

        logit = FederatedLogisticRegression(fit_intercept=False)
        logit_results = logit.fit(
            stacked_X,
            stacked_y.astype(float),
            agg_client=agg_client,
        )
        summary = logit_results.summary()

        n_covariates = X.shape[1]
        names = list(feature_names or [f"x{i}" for i in range(n_covariates)])
        coefficients = np.asarray(summary["coefficients"], dtype=float)[:n_covariates]
        std_err = np.asarray(summary["stderr"], dtype=float)[:n_covariates]
        lower_ci = np.asarray(summary["lower_ci"], dtype=float)[:n_covariates]
        upper_ci = np.asarray(summary["upper_ci"], dtype=float)[:n_covariates]
        z_scores = np.asarray(summary["z_scores"], dtype=float)[:n_covariates]
        pvalues = np.asarray(summary["pvalues"], dtype=float)[:n_covariates]
        hazard_ratios = _safe_exp(coefficients)
        hr_lower_ci = _safe_exp(lower_ci)
        hr_upper_ci = _safe_exp(upper_ci)

        results = FederatedStackedCoxRegressionResults(
            params=coefficients,
            hazard_ratios=hazard_ratios,
            std_err=std_err,
            lower_ci=lower_ci,
            upper_ci=upper_ci,
            hr_lower_ci=hr_lower_ci,
            hr_upper_ci=hr_upper_ci,
            z_scores=z_scores,
            pvalues=pvalues,
            nobs=global_nobs,
            n_events=global_n_events,
            n_stacked_rows=global_n_stacked_rows,
            n_covariates=int(n_covariates),
            df_model=int(summary["df_model"]),
            df_resid=int(summary["df_resid"]),
            r_squared_cs=float(summary["r_squared_cs"]),
            r_squared_mcf=float(summary["r_squared_mcf"]),
            ll0=float(summary["ll0"]),
            ll=float(summary["ll"]),
            aic=float(summary["aic"]),
            bic=float(summary["bic"]),
            indep_vars=names,
            time_grid_strategy=self.time_grid_strategy,
            n_time_bins_used=(
                int(len(time_bins) - 1)
                if np.isclose(np.asarray(time_bins, dtype=float).reshape(-1)[0], 0.0)
                else int(len(time_bins))
            ),
            time_bins=np.asarray(time_bins, dtype=float),
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

    @classmethod
    def _max_identifiable_time_bins(
        cls,
        *,
        n_events: int,
        n_covariates: int,
    ) -> int:
        return int(n_events) - int(n_covariates) - 1

    @classmethod
    def _recommended_max_time_bins(
        cls,
        *,
        n_events: int,
        n_covariates: int,
    ) -> int:
        max_identifiable = cls._max_identifiable_time_bins(
            n_events=n_events,
            n_covariates=n_covariates,
        )
        if max_identifiable <= 0:
            raise BadInputError(
                "cox_regression_stacked requires more observed events than covariates "
                "to fit the stacked logistic model."
            )
        return max(1, min(max_identifiable, int(n_events) // 2))

    @classmethod
    def build_distinct_event_times(
        cls,
        *,
        global_event_times: Sequence[float],
        global_max_time: float,
        max_time_bins: int | None = None,
    ) -> np.ndarray:
        event_times = np.asarray(global_event_times, dtype=float).reshape(-1)
        event_times = event_times[np.isfinite(event_times)]
        event_times = np.unique(event_times[event_times > 0])
        if event_times.size == 0:
            raise BadInputError(
                "cox_regression_stacked requires at least one observed event."
            )
        if max_time_bins is None or event_times.size <= max_time_bins:
            return event_times

        group_last_indices = _grouped_last_indices(event_times.size, max_time_bins)
        boundaries = [0.0]
        for idx in group_last_indices[:-1]:
            boundaries.append(float(np.nextafter(event_times[idx], np.inf)))
        boundaries.append(_strictly_positive_upper_edge(float(global_max_time)))
        event_times = np.asarray(boundaries, dtype=float)
        return event_times

    @classmethod
    def build_uniform_time_bins(
        cls,
        *,
        global_max_time: float,
        n_time_bins: int,
    ) -> np.ndarray:
        if n_time_bins <= 0:
            raise BadInputError("n_time_bins must be positive.")
        upper_edge = _strictly_positive_upper_edge(float(global_max_time))
        return np.linspace(0.0, upper_edge, n_time_bins + 1, dtype=float)

    def compute_time_bins(
        self,
        *,
        times: np.ndarray,
        events: np.ndarray,
        agg_client: AggregationClient,
        max_time_bins: int | None = None,
    ) -> np.ndarray:
        local_max_time = float(np.max(times))
        global_max_time = float(
            np.asarray(
                agg_client.max(np.array([local_max_time], dtype=float)),
                dtype=float,
            ).reshape(-1)[0]
        )

        if self.time_grid_strategy == "uniform":
            return self.build_uniform_time_bins(
                global_max_time=global_max_time,
                n_time_bins=min(self.n_time_bins, max_time_bins)
                if max_time_bins is not None
                else self.n_time_bins,
            )

        local_event_times = np.unique(times[events > 0]).astype(float)
        global_event_times = agg_client.union(local_event_times.tolist())
        return self.build_distinct_event_times(
            global_event_times=global_event_times,
            global_max_time=global_max_time,
            max_time_bins=max_time_bins,
        )

    @classmethod
    def stack(
        cls,
        *,
        X: np.ndarray,
        times: np.ndarray,
        events: np.ndarray,
        time_bins: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        times = _as_1d_float(times)
        events = _as_1d_float(events)
        cls._validate_fit_inputs(X, times, events)

        permutation = np.argsort(times, kind="mergesort")
        X_sorted = X[permutation]
        times_sorted = times[permutation]
        events_sorted = events[permutation]

        bins = np.asarray(time_bins, dtype=float).reshape(-1)
        if bins.size == 0:
            raise BadInputError("time grid must contain at least one value.")
        if np.isclose(bins[0], 0.0):
            cls._validate_time_bins(time_bins=bins, times=times)
            bin_start_indices = np.array(
                [
                    np.searchsorted(times_sorted, bin_start, side="left")
                    for bin_start in bins
                ],
                dtype=int,
            )
            return cls._stack_sorted_data(
                X_sorted=X_sorted,
                events_sorted=events_sorted,
                bin_start_indices=bin_start_indices,
            )

        cls._validate_event_times(event_times=bins, times=times, events=events)
        return cls._stack_at_event_times(
            X_sorted=X_sorted,
            times_sorted=times_sorted,
            events_sorted=events_sorted,
            event_times=bins,
        )

    @classmethod
    def _stack_sorted_data(
        cls,
        *,
        X_sorted: np.ndarray,
        events_sorted: np.ndarray,
        bin_start_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_obs, n_covariates = X_sorted.shape
        n_time_bins = len(bin_start_indices) - 1
        if n_time_bins <= 0:
            raise BadInputError("time_bins must define at least one interval.")

        risk_set_sizes = n_obs - bin_start_indices[:-1]
        stacked_size = int(np.sum(risk_set_sizes))
        stacked_X = np.zeros((stacked_size, n_covariates + n_time_bins), dtype=float)
        stacked_y = np.zeros(stacked_size, dtype=bool)

        offset = 0
        for bin_idx, start_idx in enumerate(bin_start_indices[:-1]):
            risk_set_size = int(risk_set_sizes[bin_idx])
            if risk_set_size <= 0:
                continue
            stop_idx = int(bin_start_indices[bin_idx + 1])
            stacked_X[offset : offset + risk_set_size, :n_covariates] = X_sorted[
                start_idx:
            ]
            stacked_X[offset : offset + risk_set_size, n_covariates + bin_idx] = 1.0

            failed_relative = np.where(events_sorted[start_idx:stop_idx] > 0)[0]
            stacked_y[offset + failed_relative] = True
            offset += risk_set_size

        return stacked_X, stacked_y

    @classmethod
    def _stack_at_event_times(
        cls,
        *,
        X_sorted: np.ndarray,
        times_sorted: np.ndarray,
        events_sorted: np.ndarray,
        event_times: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_obs, n_covariates = X_sorted.shape
        n_time_bins = len(event_times)
        if n_time_bins <= 0:
            raise BadInputError("event_times must define at least one failure time.")

        risk_set_sizes = np.array(
            [
                n_obs - np.searchsorted(times_sorted, event_time, side="left")
                for event_time in event_times
            ],
            dtype=int,
        )
        stacked_size = int(np.sum(risk_set_sizes))
        stacked_X = np.zeros((stacked_size, n_covariates + n_time_bins), dtype=float)
        stacked_y = np.zeros(stacked_size, dtype=bool)

        offset = 0
        for bin_idx, event_time in enumerate(event_times):
            start_idx = int(np.searchsorted(times_sorted, event_time, side="left"))
            risk_set_size = int(risk_set_sizes[bin_idx])
            if risk_set_size <= 0:
                continue
            risk_times = times_sorted[start_idx:]
            risk_events = events_sorted[start_idx:]
            event_mask = (risk_times == event_time) & (risk_events > 0)

            stacked_X[offset : offset + risk_set_size, :n_covariates] = X_sorted[
                start_idx:
            ]
            stacked_X[offset : offset + risk_set_size, n_covariates + bin_idx] = 1.0
            stacked_y[offset : offset + risk_set_size] = event_mask
            offset += risk_set_size

        return stacked_X, stacked_y

    @staticmethod
    def _validate_fit_inputs(
        X: np.ndarray,
        times: np.ndarray,
        events: np.ndarray,
    ) -> None:
        if X.ndim != 2:
            raise BadInputError(f"X must be 2D, got shape {X.shape}.")
        if times.shape != (X.shape[0],):
            raise BadInputError("times must have the same length as X.")
        if events.shape != (X.shape[0],):
            raise BadInputError("events must have the same length as X.")
        if X.shape[0] == 0:
            raise BadInputError(
                "cox_regression_stacked requires at least one observation."
            )
        if X.shape[1] == 0:
            raise BadInputError(
                "cox_regression_stacked requires at least one covariate."
            )
        if not np.isfinite(X).all():
            raise BadInputError("X contains NaN/Inf.")
        if not np.isfinite(times).all():
            raise BadInputError("times contains NaN/Inf.")
        if np.min(times) <= 0:
            raise BadInputError("All time values must be strictly positive.")
        _validate_binary_events(events)

    @staticmethod
    def _validate_time_bins(*, time_bins: np.ndarray, times: np.ndarray) -> None:
        bins = np.asarray(time_bins, dtype=float).reshape(-1)
        if bins.size < 2:
            raise BadInputError("time_bins must contain at least two cut points.")
        if not np.isclose(bins[0], 0.0):
            raise BadInputError("time_bins must start at 0.")
        if not np.all(np.diff(bins) > 0):
            raise BadInputError("time_bins must be strictly increasing.")
        if bins[-1] <= float(np.max(times)):
            raise BadInputError(
                "The final time bin cut point must be greater than the "
                "maximum time value."
            )

    @staticmethod
    def _validate_event_times(
        *,
        event_times: np.ndarray,
        times: np.ndarray,
        events: np.ndarray,
    ) -> None:
        distinct = np.asarray(event_times, dtype=float).reshape(-1)
        if distinct.size < 1:
            raise BadInputError("event_times must contain at least one failure time.")
        if not np.all(np.diff(distinct) > 0):
            raise BadInputError("event_times must be strictly increasing.")
        if np.min(distinct) <= 0:
            raise BadInputError("event_times must be strictly positive.")
