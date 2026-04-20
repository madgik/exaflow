from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from exaflow.algorithms.federated.mixed_effects.common import apply_weights
from exaflow.algorithms.federated.mixed_effects.common import build_local_hist
from exaflow.algorithms.federated.mixed_effects.common import pack_upper_triangle
from exaflow.algorithms.federated.mixed_effects.common import (
    reml_grad_logscale_from_summaries,
)
from exaflow.algorithms.federated.mixed_effects.common import (
    reml_objective_from_summaries,
)
from exaflow.algorithms.federated.mixed_effects.common import unpack_upper_triangle
from exaflow.algorithms.federated.mixed_effects.common import validate_inputs
from exaflow.algorithms.federated.mixed_effects.lmm_legacy import ALPHA
from exaflow.algorithms.federated.mixed_effects.lmm_legacy import FederatedLMMResults
from exaflow.algorithms.federated.mixed_effects.lmm_legacy import _LMMRemlAggregator
from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.utils.agg_client import AggregationClient
from exaflow.algorithms.federated.utils.interfaces import FederatedEstimator


@dataclass(frozen=True)
class _LMMClusterStats:
    label: object
    n_obs: int
    xtx: np.ndarray
    xty: np.ndarray
    yty: float
    x_sum: np.ndarray
    y_sum: float


@dataclass(frozen=True)
class _LMMProfileState:
    beta: np.ndarray
    a_inv: np.ndarray
    sxx: np.ndarray
    syy: float
    ll_reml: float


def _solve_gls_system(
    sxx: np.ndarray,
    sxy: np.ndarray,
    *,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    p = sxx.shape[0]
    a = sxx + ridge * np.eye(p, dtype=sxx.dtype)
    eye = np.eye(p, dtype=sxx.dtype)
    try:
        chol = np.linalg.cholesky(a)
        beta = np.linalg.solve(chol.T, np.linalg.solve(chol, sxy))
        a_inv = np.linalg.solve(chol.T, np.linalg.solve(chol, eye))
        return beta, a_inv
    except np.linalg.LinAlgError:
        try:
            beta = np.linalg.solve(a, sxy)
            a_inv = np.linalg.solve(a, eye)
            return beta, a_inv
        except np.linalg.LinAlgError:
            a_inv = np.linalg.pinv(a)
            beta = a_inv @ sxy
            return beta, a_inv


def _random_intercept_scalars(
    n_obs: int,
    sigma2: float,
    sigma_u2: float,
) -> tuple[float, float, float]:
    inv_sigma2 = 1.0 / sigma2
    alpha = sigma_u2 / (sigma2 * (sigma2 + n_obs * sigma_u2))
    ones_gain = inv_sigma2 - alpha * n_obs
    return inv_sigma2, alpha, ones_gain


def _prepare_cluster_stats(
    X: np.ndarray,
    y: np.ndarray,
    center_ids: np.ndarray,
    w: np.ndarray | None = None,
) -> tuple[list[_LMMClusterStats], np.ndarray, np.ndarray]:
    Xw, yw = apply_weights(X, y, w)
    centers, inverse = np.unique(center_ids, return_inverse=True)

    cluster_stats: list[_LMMClusterStats] = []
    cluster_sizes = np.zeros(centers.shape[0], dtype=int)
    cluster_sums = np.zeros(centers.shape[0], dtype=float)
    for idx, label in enumerate(centers):
        mask = inverse == idx
        xj = Xw[mask]
        yj = yw[mask]
        nj = int(xj.shape[0])
        if nj == 0:
            continue
        cluster_sizes[idx] = nj
        cluster_sums[idx] = float(np.sum(yj))
        cluster_stats.append(
            _LMMClusterStats(
                label=label,
                n_obs=nj,
                xtx=xj.T @ xj,
                xty=xj.T @ yj,
                yty=float(yj @ yj),
                x_sum=np.sum(xj, axis=0),
                y_sum=float(np.sum(yj)),
            )
        )
    return cluster_stats, cluster_sizes, cluster_sums


class FederatedLMM(FederatedEstimator):
    def __init__(
        self,
        *,
        fit_intercept: bool = True,
        max_iter: int = 80,
        min_iter: int = 10,
        tol: float = 1e-8,
        ridge: float = 1e-8,
        lower_bound: float = 1e-6,
        upper_bound: float = 1e6,
        reg_lambda: float = 1e-2,
        init_sigma2: float = 1.0,
        init_sigma_u2: float = 0.5,
        use_rough_init: bool = True,
        return_history: bool = False,
    ) -> None:
        if max_iter <= 0:
            raise BadInputError("max_iter must be positive.")
        if tol <= 0:
            raise BadInputError("tol must be positive.")
        if min_iter <= 0:
            raise BadInputError("min_iter must be positive.")
        if min_iter > max_iter:
            raise BadInputError("min_iter cannot be greater than max_iter.")
        if ridge < 0:
            raise BadInputError("ridge must be non-negative.")
        if lower_bound <= 0:
            raise BadInputError("lower_bound must be positive.")
        if upper_bound <= lower_bound:
            raise BadInputError("upper_bound must be greater than lower_bound.")
        if reg_lambda < 0:
            raise BadInputError("reg_lambda must be non-negative.")
        if init_sigma2 <= 0 or init_sigma_u2 <= 0:
            raise BadInputError("init_sigma2 and init_sigma_u2 must be positive.")
        if (
            init_sigma2 < lower_bound
            or init_sigma_u2 < lower_bound
            or init_sigma2 > upper_bound
            or init_sigma_u2 > upper_bound
        ):
            raise BadInputError(
                "init_sigma2 and init_sigma_u2 must be within [lower_bound, upper_bound]."
            )

        self.fit_intercept = fit_intercept
        self.max_iter = int(max_iter)
        self.min_iter = int(min_iter)
        self.tol = float(tol)
        self.ridge = float(ridge)
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self.reg_lambda = float(reg_lambda)
        self.init_sigma2 = float(init_sigma2)
        self.init_sigma_u2 = float(init_sigma_u2)
        self.use_rough_init = bool(use_rough_init)
        self.return_history = bool(return_history)
        self.results: FederatedLMMResults | None = None

    @staticmethod
    def _add_intercept(X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_obs = X.shape[0]
        return np.hstack([np.ones((n_obs, 1), dtype=float), X])

    def _rough_variance_init(
        self,
        y: np.ndarray,
        center_ids: np.ndarray,
        *,
        agg_client: AggregationClient,
        cluster_sizes: np.ndarray,
        cluster_sums: np.ndarray,
        cluster_labels: np.ndarray,
    ) -> tuple[float, float]:
        n_obs = int(
            np.asarray(
                agg_client.sum(np.array([float(y.shape[0])], dtype=float)),
                dtype=float,
            ).reshape(-1)[0]
        )
        sum_y = float(
            np.asarray(agg_client.sum(np.array([float(y.sum())], dtype=float))).reshape(
                -1
            )[0]
        )
        sum_sq_y = float(
            np.asarray(
                agg_client.sum(np.array([float(np.dot(y, y))], dtype=float))
            ).reshape(-1)[0]
        )
        if n_obs <= 1:
            return self.init_sigma2, self.init_sigma_u2

        global_mean = sum_y / n_obs
        total_var = max(sum_sq_y / n_obs - global_mean * global_mean, self.lower_bound)

        centers = list(agg_client.union(cluster_labels.tolist()))
        if not centers:
            return max(total_var, self.lower_bound), self.lower_bound

        idx_map = {cid: i for i, cid in enumerate(centers)}
        local_counts = np.zeros(len(centers), dtype=float)
        local_sums = np.zeros(len(centers), dtype=float)
        for cid, count, sum_yj in zip(
            cluster_labels,
            cluster_sizes,
            cluster_sums,
            strict=True,
        ):
            local_idx = idx_map[cid]
            local_counts[local_idx] = float(count)
            local_sums[local_idx] = float(sum_yj)

        counts = np.asarray(agg_client.sum(local_counts), dtype=float)
        sums = np.asarray(agg_client.sum(local_sums), dtype=float)
        valid = counts > 0
        if not np.any(valid):
            return max(total_var, self.lower_bound), self.lower_bound

        means = np.zeros_like(sums)
        means[valid] = sums[valid] / counts[valid]
        between = float(
            np.sum(counts[valid] * (means[valid] - global_mean) ** 2) / n_obs
        )
        between = min(max(between, self.lower_bound), self.upper_bound)
        within = min(max(total_var - between, self.lower_bound), self.upper_bound)
        return within, between

    def _collect_global_hist(
        self,
        cluster_stats: list[_LMMClusterStats],
        agg_client: AggregationClient,
    ) -> np.ndarray:
        local_max, _ = build_local_hist([stat.n_obs for stat in cluster_stats])
        global_max = int(
            np.asarray(
                agg_client.max(np.array([float(local_max)], dtype=float)),
                dtype=float,
            ).reshape(-1)[0]
        )
        if global_max <= 0:
            return np.zeros(0, dtype=np.int64)
        _, local_hist = build_local_hist(
            [stat.n_obs for stat in cluster_stats],
            K=global_max,
        )
        return np.asarray(agg_client.sum(local_hist.astype(float)), dtype=float).astype(
            np.int64
        )

    def _local_phase_a_vector(
        self,
        cluster_stats: list[_LMMClusterStats],
        *,
        sigma2: float,
        sigma_u2: float,
        p: int,
    ) -> np.ndarray:
        sxx = np.zeros((p, p), dtype=float)
        sxy = np.zeros(p, dtype=float)
        syy = 0.0
        for stat in cluster_stats:
            inv_sigma2, alpha, _ = _random_intercept_scalars(
                stat.n_obs,
                sigma2,
                sigma_u2,
            )
            sxx += inv_sigma2 * stat.xtx - alpha * np.outer(stat.x_sum, stat.x_sum)
            sxy += inv_sigma2 * stat.xty - alpha * stat.x_sum * stat.y_sum
            syy += inv_sigma2 * stat.yty - alpha * stat.y_sum * stat.y_sum
        packed_sxx = np.asarray(pack_upper_triangle(sxx), dtype=float)
        return np.concatenate([packed_sxx, sxy, np.array([syy], dtype=float)])

    def _profile_eval(
        self,
        cluster_stats: list[_LMMClusterStats],
        *,
        sigma2: float,
        sigma_u2: float,
        p: int,
        global_hist: np.ndarray,
        n_obs: int,
        agg_client: AggregationClient,
    ) -> _LMMProfileState:
        phase_a_local = self._local_phase_a_vector(
            cluster_stats,
            sigma2=sigma2,
            sigma_u2=sigma_u2,
            p=p,
        )
        phase_a_global = np.asarray(agg_client.sum(phase_a_local), dtype=float)
        packed_len = (p * (p + 1)) // 2
        sxx = unpack_upper_triangle(phase_a_global[:packed_len], p)
        sxy = phase_a_global[packed_len : packed_len + p]
        syy = float(phase_a_global[-1])
        beta, a_inv = _solve_gls_system(sxx, sxy, ridge=self.ridge)
        ll_reml = reml_objective_from_summaries(
            sxx,
            syy,
            beta,
            sigma2=sigma2,
            sigma_u2=sigma_u2,
            hist=global_hist,
            n_obs=n_obs,
            p=p,
        )
        return _LMMProfileState(
            beta=beta,
            a_inv=a_inv,
            sxx=sxx,
            syy=syy,
            ll_reml=float(ll_reml),
        )

    def _local_phase_b_vector(
        self,
        cluster_stats: list[_LMMClusterStats],
        *,
        sigma2: float,
        sigma_u2: float,
        beta: np.ndarray,
        p: int,
    ) -> np.ndarray:
        q1 = 0.0
        q2 = 0.0
        t0 = 0.0
        t1 = 0.0
        b = np.zeros((p, p), dtype=float)
        for stat in cluster_stats:
            inv_sigma2, alpha, ones_gain = _random_intercept_scalars(
                stat.n_obs,
                sigma2,
                sigma_u2,
            )
            sum_r = stat.y_sum - float(stat.x_sum @ beta)
            rtr = float(
                stat.yty
                - 2.0 * np.dot(beta, stat.xty)
                + beta @ (stat.xtx @ beta)
            )
            q1 += (
                inv_sigma2 * inv_sigma2 * rtr
                + (-2.0 * inv_sigma2 * alpha + alpha * alpha * stat.n_obs)
                * sum_r
                * sum_r
            )
            q2 += (ones_gain * sum_r) ** 2
            t0 += stat.n_obs * (inv_sigma2 - alpha)
            t1 += stat.n_obs * ones_gain
            xv = ones_gain * stat.x_sum
            b += np.outer(xv, xv)
        packed_b = np.asarray(pack_upper_triangle(b), dtype=float)
        return np.concatenate(
            [np.array([q1, q2, t0, t1], dtype=float), packed_b],
        )

    def _gradient_logscale(
        self,
        cluster_stats: list[_LMMClusterStats],
        *,
        sigma2: float,
        sigma_u2: float,
        beta: np.ndarray,
        a_inv: np.ndarray,
        p: int,
        agg_client: AggregationClient,
    ) -> np.ndarray:
        phase_b_local = self._local_phase_b_vector(
            cluster_stats,
            sigma2=sigma2,
            sigma_u2=sigma_u2,
            beta=beta,
            p=p,
        )
        phase_b_global = np.asarray(agg_client.sum(phase_b_local), dtype=float)
        q1, q2, t0, t1 = phase_b_global[:4]
        b = unpack_upper_triangle(phase_b_global[4:], p)
        return reml_grad_logscale_from_summaries(
            q1=q1,
            q2=q2,
            t0=t0,
            t1=t1,
            b=b,
            a_inv=a_inv,
            p=p,
            sigma2=sigma2,
            sigma_u2=sigma_u2,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        center_ids: np.ndarray,
        agg_client: AggregationClient,
        w: np.ndarray | None = None,
    ) -> FederatedLMMResults:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        center_ids = np.asarray(center_ids)
        if self.fit_intercept:
            X = self._add_intercept(X)
        validate_inputs(X, y, center_ids, w)

        n_obs = int(
            np.asarray(
                agg_client.sum(np.array([float(X.shape[0])], dtype=float)),
                dtype=float,
            ).reshape(-1)[0]
        )
        if n_obs <= X.shape[1]:
            raise BadInputError(
                "LMM cannot run because observations are fewer than predictors."
            )

        cluster_stats, cluster_sizes, cluster_sums = _prepare_cluster_stats(
            X,
            y,
            center_ids,
            w,
        )
        cluster_labels = np.asarray([stat.label for stat in cluster_stats], dtype=object)
        global_hist = self._collect_global_hist(cluster_stats, agg_client)

        p = X.shape[1]
        beta = np.zeros(p, dtype=float)
        if self.use_rough_init:
            sigma2, sigma_u2 = self._rough_variance_init(
                y,
                center_ids,
                agg_client=agg_client,
                cluster_sizes=cluster_sizes,
                cluster_sums=cluster_sums,
                cluster_labels=cluster_labels,
            )
        else:
            sigma2 = float(self.init_sigma2)
            sigma_u2 = float(self.init_sigma_u2)

        converged = False
        history: list[dict[str, float]] = []
        final_profile: _LMMProfileState | None = None

        for it in range(1, self.max_iter + 1):
            current_profile = self._profile_eval(
                cluster_stats,
                sigma2=sigma2,
                sigma_u2=sigma_u2,
                p=p,
                global_hist=global_hist,
                n_obs=n_obs,
                agg_client=agg_client,
            )
            grad = self._gradient_logscale(
                cluster_stats,
                sigma2=sigma2,
                sigma_u2=sigma_u2,
                beta=current_profile.beta,
                a_inv=current_profile.a_inv,
                p=p,
                agg_client=agg_client,
            )

            phi0 = np.array([np.log(sigma2), np.log(sigma_u2)], dtype=float)
            direction = grad / (1.0 + self.reg_lambda)
            gd = float(grad @ direction)
            log_lo = float(np.log(self.lower_bound))
            log_hi = float(np.log(self.upper_bound))

            sigma2_new = sigma2
            sigma_u2_new = sigma_u2
            selected_profile = current_profile
            improved = False

            if np.isfinite(gd) and gd > 0.0:
                step = 1.0
                for _ in range(10):
                    phi_try = np.clip(phi0 + step * direction, log_lo, log_hi)
                    sigma2_try = float(np.exp(phi_try[0]))
                    sigma_u2_try = float(np.exp(phi_try[1]))
                    trial_profile = self._profile_eval(
                        cluster_stats,
                        sigma2=sigma2_try,
                        sigma_u2=sigma_u2_try,
                        p=p,
                        global_hist=global_hist,
                        n_obs=n_obs,
                        agg_client=agg_client,
                    )
                    if np.isfinite(trial_profile.ll_reml) and (
                        trial_profile.ll_reml
                        >= current_profile.ll_reml + 1e-4 * step * gd
                    ):
                        sigma2_new = sigma2_try
                        sigma_u2_new = sigma_u2_try
                        selected_profile = trial_profile
                        improved = True
                        break
                    step *= 0.5

            delta = max(
                float(np.linalg.norm(selected_profile.beta - beta))
                / (1.0 + float(np.linalg.norm(beta))),
                abs(sigma2_new - sigma2) / (1.0 + sigma2),
                abs(sigma_u2_new - sigma_u2) / (1.0 + sigma_u2),
            )
            history.append(
                {
                    "iter": float(it),
                    "delta": float(delta),
                    "sigma2": float(sigma2_new),
                    "sigma_u2": float(sigma_u2_new),
                    "ll_reml": float(selected_profile.ll_reml),
                    "improved": float(improved),
                }
            )

            beta = selected_profile.beta
            sigma2 = sigma2_new
            sigma_u2 = sigma_u2_new
            final_profile = selected_profile

            if it >= self.min_iter and delta < self.tol:
                converged = True
                break

        if final_profile is None:  # pragma: no cover - defensive
            raise RuntimeError("LMMV2 finished without a profile state.")

        cov_params = final_profile.a_inv * sigma2
        bse = np.sqrt(np.maximum(np.diag(cov_params), 0.0))
        df_model = int(p - 1 if self.fit_intercept else p)
        df_resid = int(n_obs - p)

        if df_resid > 0:
            tvalues = np.divide(
                beta,
                bse,
                out=np.zeros_like(beta, dtype=float),
                where=bse != 0,
            )
            pvalues = stats.t.sf(np.abs(tvalues), df=df_resid) * 2.0
            t_crit = stats.t.ppf(1.0 - ALPHA / 2.0, df=df_resid)
            conf_low = beta - t_crit * bse
            conf_high = beta + t_crit * bse
        else:
            tvalues = np.full_like(beta, np.nan, dtype=float)
            pvalues = np.full_like(beta, np.nan, dtype=float)
            conf_low = np.full_like(beta, np.nan, dtype=float)
            conf_high = np.full_like(beta, np.nan, dtype=float)

        n_groups = int(np.sum(global_hist)) if global_hist.size else 0
        k_params = p + 2
        aic = float(2.0 * k_params - 2.0 * final_profile.ll_reml)
        bic = float(np.log(max(n_obs, 1)) * k_params - 2.0 * final_profile.ll_reml)

        results = FederatedLMMResults(
            params=beta,
            bse=bse,
            tvalues=tvalues,
            pvalues=pvalues,
            conf_int_low=conf_low,
            conf_int_high=conf_high,
            cov_params=cov_params,
            sigma2=sigma2,
            sigma_u2=sigma_u2,
            nobs=n_obs,
            n_groups=n_groups,
            df_model=df_model,
            df_resid=df_resid,
            ll_reml=float(final_profile.ll_reml),
            aic=aic,
            bic=bic,
            converged=converged,
            n_iter=len(history),
            fit_intercept=self.fit_intercept,
            history=history if self.return_history else None,
        )
        self.results = results
        return results
