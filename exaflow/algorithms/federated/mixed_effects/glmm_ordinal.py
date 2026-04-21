from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from exaflow.algorithms.federated.mixed_effects.common import pack_upper_triangle
from exaflow.algorithms.federated.mixed_effects.glmm_ordinal_legacy import (
    FederatedGLMMOrdinalResults,
)
from exaflow.algorithms.federated.mixed_effects.glmm_ordinal_legacy import (
    _GLMMAggregator,
)
from exaflow.algorithms.federated.mixed_effects.glmm_ordinal_legacy import (
    _build_kappas_from_theta,
)
from exaflow.algorithms.federated.mixed_effects.glmm_ordinal_legacy import (
    _ordinal_pi_and_grads,
)
from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.utils.agg_client import AggregationClient
from exaflow.algorithms.federated.utils.interfaces import FederatedEstimator


@dataclass(frozen=True)
class _OrdinalClusterData:
    center_id: object
    X: np.ndarray
    y: np.ndarray


def _prepare_ordinal_clusters(
    X: np.ndarray,
    y: np.ndarray,
    center_ids: np.ndarray,
) -> list[_OrdinalClusterData]:
    centers, inverse = np.unique(center_ids, return_inverse=True)
    clusters: list[_OrdinalClusterData] = []
    for idx, center_id in enumerate(centers):
        mask = inverse == idx
        if not np.any(mask):
            continue
        clusters.append(
            _OrdinalClusterData(
                center_id=center_id,
                X=np.asarray(X[mask], dtype=float),
                y=np.asarray(y[mask], dtype=int),
            )
        )
    return clusters


def _mode_u_for_center_with_k_warm(
    eta_base: np.ndarray,
    y: np.ndarray,
    kappas: np.ndarray,
    sigma_u2: float,
    *,
    init_u: float,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> tuple[float, float]:
    eta_base = np.asarray(eta_base, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=int).reshape(-1)
    if sigma_u2 <= 0.0:
        raise BadInputError("sigma_u2 must be > 0.")

    u = float(init_u)
    h_sum = -1.0 / sigma_u2
    for _ in range(max_iter):
        g_sum = 0.0
        h_sum = -1.0 / sigma_u2
        for i in range(y.size):
            _, dlog_deta, d2log_deta2, _ = _ordinal_pi_and_grads(
                float(eta_base[i] + u),
                int(y[i]),
                kappas,
            )
            g_sum += dlog_deta
            h_sum += d2log_deta2

        if h_sum >= -1e-10:
            h_sum = -1e-10
        step = g_sum / h_sum
        u_new = u - step
        if abs(u_new - u) > 5.0:
            u_new = u + np.sign(u_new - u) * 5.0
        if abs(u_new - u) < tol:
            u = u_new
            break
        u = u_new

    huu = -h_sum
    return float(u), float(huu)


def _suffix_sums(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values.copy()
    return np.flip(np.cumsum(np.flip(values)))


class FederatedGLMMOrdinal(FederatedEstimator):
    def __init__(
        self,
        *,
        K: int,
        fit_intercept: bool = True,
        max_iters: int = 50,
        ridge: float = 1e-6,
        tol_theta: float = 1e-6,
        tol_score: float = 1e-4,
        max_step_norm: float = 5.0,
        clip_log_sigma_bounds: tuple[float, float] = (np.log(1e-8), np.log(1e3)),
        log_sigma_u2_init: float = np.log(0.3),
        return_history: bool = False,
        mode_tol: float = 1e-8,
        mode_max_iter: int = 50,
    ) -> None:
        if K < 2:
            raise BadInputError("K must be >= 2.")
        if max_iters <= 0:
            raise BadInputError("max_iters must be positive.")
        if ridge < 0:
            raise BadInputError("ridge must be non-negative.")
        if tol_theta <= 0 or tol_score <= 0:
            raise BadInputError("tol_theta and tol_score must be positive.")
        if max_step_norm <= 0:
            raise BadInputError("max_step_norm must be positive.")
        if mode_tol <= 0:
            raise BadInputError("mode_tol must be positive.")
        if mode_max_iter <= 0:
            raise BadInputError("mode_max_iter must be positive.")

        self.K = int(K)
        self.fit_intercept = bool(fit_intercept)
        self.max_iters = int(max_iters)
        self.ridge = float(ridge)
        self.tol_theta = float(tol_theta)
        self.tol_score = float(tol_score)
        self.max_step_norm = float(max_step_norm)
        self.clip_log_sigma_bounds = clip_log_sigma_bounds
        self.log_sigma_u2_init = float(log_sigma_u2_init)
        self.return_history = bool(return_history)
        self.mode_tol = float(mode_tol)
        self.mode_max_iter = int(mode_max_iter)
        self.results: FederatedGLMMOrdinalResults | None = None

    @staticmethod
    def _add_intercept(X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.hstack([np.ones((X.shape[0], 1), dtype=float), X])

    def _site_derivatives_from_clusters(
        self,
        clusters: list[_OrdinalClusterData],
        theta: np.ndarray,
        *,
        p: int,
        mode_state: np.ndarray,
    ) -> dict[str, np.ndarray | float | int]:
        kappas, log_su2 = _build_kappas_from_theta(theta, p, self.K)
        sigma_u2 = float(np.exp(log_su2))
        q = p + 1 + (self.K - 1)

        score = np.zeros(q, dtype=float)
        h = np.zeros((q, q), dtype=float)
        beta = np.asarray(theta[:p], dtype=float)
        gammas = np.asarray(theta[p + 2 : p + 1 + (self.K - 1)], dtype=float)
        exp_g = np.exp(gammas)

        sum_log_huu = 0.0
        for idx, cluster in enumerate(clusters):
            xj = cluster.X
            yj = cluster.y
            eta0 = xj @ beta

            u_star, huu = _mode_u_for_center_with_k_warm(
                eta0,
                yj,
                kappas,
                sigma_u2,
                init_u=float(mode_state[idx]),
                max_iter=self.mode_max_iter,
                tol=self.mode_tol,
            )
            mode_state[idx] = u_star
            sum_log_huu += np.log(huu)

            nj = yj.size
            dlog_eta = np.empty(nj, dtype=float)
            dlog_k = np.zeros((nj, self.K - 1), dtype=float)
            for row_idx in range(nj):
                eta = float(eta0[row_idx] + u_star)
                _, dlog_deta, _, dlog_dk = _ordinal_pi_and_grads(
                    eta,
                    int(yj[row_idx]),
                    kappas,
                )
                dlog_eta[row_idx] = dlog_deta
                dlog_k[row_idx] = np.asarray(dlog_dk, dtype=float)

            grad_kappa_sum = np.sum(dlog_k, axis=0)
            score[:p] += xj.T @ dlog_eta
            score[p] += 0.5 * (u_star * u_star / sigma_u2 - 1.0)
            if self.K > 1:
                score[p + 1] += grad_kappa_sum[0]
            if self.K > 2:
                score[p + 2 : p + 1 + (self.K - 1)] += exp_g * _suffix_sums(
                    grad_kappa_sum[1:]
                )

            g_beta = dlog_eta[:, None] * xj
            blocks = [g_beta, np.zeros((nj, 1), dtype=float)]
            if self.K > 1:
                blocks.append(dlog_k[:, 0:1])
            if self.K > 2:
                gamma_tail = np.flip(
                    np.cumsum(np.flip(dlog_k[:, 1:], axis=1), axis=1),
                    axis=1,
                )
                blocks.append(gamma_tail * exp_g)
            g_mat = np.hstack(blocks)
            h -= g_mat.T @ g_mat

        h[p, p] += -0.5 * len(clusters)
        h = 0.5 * (h + h.T)
        return {
            "q": int(q),
            "score": score.astype(float),
            "h_packed": np.asarray(pack_upper_triangle(h.astype(float)), dtype=float),
            "n_centers": int(len(clusters)),
            "sum_log_huu": float(sum_log_huu),
        }

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        center_ids: np.ndarray,
        agg_client: AggregationClient,
    ) -> FederatedGLMMOrdinalResults:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).reshape(-1)
        center_ids = np.asarray(center_ids)
        if self.fit_intercept:
            X = self._add_intercept(X)
        if X.shape[0] != y.shape[0] or y.shape[0] != center_ids.shape[0]:
            raise BadInputError("X, y, center_ids must have same length.")
        if y.min() < 0 or y.max() >= self.K:
            raise BadInputError(f"y must be in [0, {self.K - 1}].")

        n_obs = int(
            np.asarray(
                agg_client.sum(np.array([float(X.shape[0])], dtype=float))
            ).reshape(-1)[0]
        )
        try:
            n_groups = len(agg_client.union(center_ids.tolist()))
        except Exception:
            n_groups = len(np.unique(center_ids))

        p = X.shape[1]
        q = p + 1 + (self.K - 1)
        theta = np.concatenate(
            [
                np.zeros(p, dtype=float),
                np.array([self.log_sigma_u2_init], dtype=float),
                np.array([0.0], dtype=float),
                np.zeros(max(0, self.K - 2), dtype=float),
            ]
        )

        clusters = _prepare_ordinal_clusters(X, y, center_ids)
        mode_state = np.zeros(len(clusters), dtype=float)
        history: list[dict[str, float]] = []
        converged = False
        for it in range(1, self.max_iters + 1):
            site = self._site_derivatives_from_clusters(
                clusters,
                theta,
                p=p,
                mode_state=mode_state,
            )
            score_sum = np.asarray(agg_client.sum(site["score"]), dtype=float).reshape(
                q
            )
            h_packed_sum = np.asarray(agg_client.sum(site["h_packed"]), dtype=float)

            agg = _GLMMAggregator(q, log_sigma_index=p)
            agg.accumulate({"score": score_sum, "h_packed": h_packed_sum})
            theta_new = agg.newton_update(
                theta,
                ridge=self.ridge,
                max_tries=6,
                max_step_norm=self.max_step_norm,
                clip_log_sigma_bounds=self.clip_log_sigma_bounds,
            )

            score_norm = float(np.linalg.norm(agg.s))
            dtheta_max = float(np.max(np.abs(theta_new - theta)))
            history.append(
                {"iter": float(it), "score_norm": score_norm, "dtheta_max": dtheta_max}
            )

            if _GLMMAggregator.converged(
                theta,
                theta_new,
                agg.s,
                tol_theta=self.tol_theta,
                tol_score=self.tol_score,
            ):
                theta = theta_new
                converged = True
                break
            theta = theta_new

        kappas, log_su2 = _build_kappas_from_theta(theta, p, self.K)
        results = FederatedGLMMOrdinalResults(
            theta=theta,
            params=theta[:p],
            sigma_u2=float(np.exp(log_su2)),
            cutpoints=kappas,
            nobs=n_obs,
            n_groups=n_groups,
            converged=converged,
            n_iter=len(history),
            fit_intercept=self.fit_intercept,
            K=self.K,
            history=history if self.return_history else None,
        )
        self.results = results
        return results
