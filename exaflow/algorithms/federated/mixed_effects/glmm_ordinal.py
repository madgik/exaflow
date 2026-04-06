from __future__ import annotations

from typing import Any

import numpy as np

from exaflow.algorithms.federated.mixed_effects.common import pack_upper_triangle
from exaflow.algorithms.federated.mixed_effects.common import unpack_upper_triangle
from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.utils.agg_client import AggregationClient
from exaflow.algorithms.federated.utils.interfaces import FederatedEstimator
from exaflow.algorithms.federated.utils.interfaces import FederatedEstimatorResults

_SIG_CLIP = 40.0


def _sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    zc = np.clip(z, -_SIG_CLIP, _SIG_CLIP)
    return 1.0 / (1.0 + np.exp(-zc))


def _build_kappas_from_theta(
    theta: np.ndarray,
    p: int,
    K: int,
) -> tuple[np.ndarray, float]:
    theta = np.asarray(theta, dtype=float).reshape(-1)
    if K < 2:
        raise BadInputError(f"K must be >= 2 (got {K}).")
    expected_len = p + 1 + (K - 1)
    if theta.shape[0] != expected_len:
        raise BadInputError(
            f"theta length {theta.shape[0]} != p+1+(K-1)={expected_len} (p={p}, K={K})."
        )

    log_su2 = float(theta[p])
    kappa1 = float(theta[p + 1])
    gammas = np.asarray(theta[p + 2 : p + 1 + (K - 1)], dtype=float)

    gaps = np.exp(gammas)
    kappas = np.empty(K - 1, dtype=float)
    kappas[0] = kappa1
    for i in range(1, K - 1):
        kappas[i] = kappas[i - 1] + gaps[i - 1]
    return kappas, log_su2


def _ordinal_pi_and_grads(
    eta: float,
    y: int,
    kappas: np.ndarray,
) -> tuple[float, float, float, np.ndarray]:
    kappas = np.asarray(kappas, dtype=float)
    K = kappas.shape[0] + 1
    if not (0 <= y <= K - 1):
        raise BadInputError(f"y must be in [0, {K - 1}], got {y}.")

    a = kappas - eta
    f_inner = np.asarray(_sigmoid(a), dtype=float)
    fp_inner = -f_inner * (1.0 - f_inner)
    fpp_inner = f_inner * (1.0 - f_inner) * (1.0 - 2.0 * f_inner)

    f = np.empty(K + 1, dtype=float)
    f[0] = 0.0
    f[1:K] = f_inner
    f[K] = 1.0

    fp = np.zeros(K + 1, dtype=float)
    fp[1:K] = fp_inner

    fpp = np.zeros(K + 1, dtype=float)
    fpp[1:K] = fpp_inner

    pi = float(f[y + 1] - f[y])
    pi = float(np.clip(pi, 1e-12, 1.0))
    dpi = float(fp[y + 1] - fp[y])
    d2pi = float(fpp[y + 1] - fpp[y])

    dlog = dpi / pi
    d2log = d2pi / pi - (dpi * dpi) / (pi * pi)

    dlog_dk = np.zeros(K - 1, dtype=float)
    if y >= 1:
        d_fy_dk = f[y] * (1.0 - f[y])
        dlog_dk[y - 1] += (-d_fy_dk) / pi
    if y + 1 <= K - 1:
        d_fyp1_dk = f[y + 1] * (1.0 - f[y + 1])
        dlog_dk[y] += d_fyp1_dk / pi
    return pi, dlog, d2log, dlog_dk


def _mode_u_for_center_with_k(
    eta_base: np.ndarray,
    y: np.ndarray,
    kappas: np.ndarray,
    sigma_u2: float,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> tuple[float, float]:
    eta_base = np.asarray(eta_base, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=int).reshape(-1)
    if sigma_u2 <= 0.0:
        raise BadInputError("sigma_u2 must be > 0.")

    u = 0.0
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


class _GLMMAggregator:
    def __init__(self, q: int, *, log_sigma_index: int):
        self.q = int(q)
        self.log_sigma_index = int(log_sigma_index)
        self.reset()

    def reset(self) -> None:
        self.s = np.zeros(self.q, dtype=float)
        self.h = np.zeros((self.q, self.q), dtype=float)

    def accumulate(self, payload: dict[str, Any]) -> None:
        score = np.asarray(payload["score"], dtype=float).reshape(-1)
        if score.shape[0] != self.q:
            raise BadInputError(f"score length {score.shape[0]} != q={self.q}")
        hn = unpack_upper_triangle(np.asarray(payload["h_packed"], dtype=float), self.q)
        self.s += score
        self.h += hn

    def newton_update(
        self,
        theta: np.ndarray,
        *,
        ridge: float,
        max_tries: int,
        max_step_norm: float,
        clip_log_sigma_bounds: tuple[float, float],
    ) -> np.ndarray:
        theta = np.asarray(theta, dtype=float).reshape(-1)
        h_sym = 0.5 * (self.h + self.h.T)
        eye = np.eye(self.q, dtype=float)
        ridge = float(max(ridge, 0.0))
        h_r = h_sym + ridge * eye
        delta = None
        for _ in range(max_tries):
            try:
                delta = np.linalg.solve(h_r, self.s)
                break
            except np.linalg.LinAlgError:
                ridge *= 10.0
                h_r = h_sym + ridge * eye
        if delta is None:
            # Mirror the binary GLMM fallback to stay robust on collinear or
            # near-singular real-world design matrices.
            delta = np.linalg.pinv(h_sym + max(ridge, 1e-6) * eye) @ self.s
        nrm = float(np.linalg.norm(delta))
        if nrm > max_step_norm and nrm > 0.0:
            delta *= max_step_norm / nrm
        theta_new = theta - delta
        lo, hi = clip_log_sigma_bounds
        theta_new[self.log_sigma_index] = float(
            np.clip(theta_new[self.log_sigma_index], lo, hi)
        )
        return theta_new

    @staticmethod
    def converged(
        theta: np.ndarray,
        theta_new: np.ndarray,
        score: np.ndarray,
        *,
        tol_theta: float,
        tol_score: float,
    ) -> bool:
        dtheta_max = float(np.max(np.abs(theta_new - theta)))
        snorm = float(np.linalg.norm(score))
        return (dtheta_max < tol_theta) and (snorm < tol_score)


class FederatedGLMMOrdinalResults(FederatedEstimatorResults):
    nobs: int

    def __init__(
        self,
        *,
        theta: np.ndarray,
        params: np.ndarray,
        sigma_u2: float,
        cutpoints: np.ndarray,
        nobs: int,
        n_groups: int,
        converged: bool,
        n_iter: int,
        fit_intercept: bool,
        K: int,
        history: list[dict[str, float]] | None = None,
    ) -> None:
        self.theta = np.asarray(theta, dtype=float)
        self.params = np.asarray(params, dtype=float)
        self.sigma_u2 = float(sigma_u2)
        self.cutpoints = np.asarray(cutpoints, dtype=float)
        self.nobs = int(nobs)
        self.n_groups = int(n_groups)
        self.converged = bool(converged)
        self.n_iter = int(n_iter)
        self.fit_intercept = bool(fit_intercept)
        self.K = int(K)
        self.history = history

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.fit_intercept:
            X = FederatedGLMMOrdinal._add_intercept(X)
        eta = X @ self.params
        n = eta.shape[0]
        K = self.K
        out = np.zeros((n, K), dtype=float)
        kappa = self.cutpoints
        for i in range(n):
            ei = float(eta[i])
            f = np.zeros(K + 1, dtype=float)
            f[0] = 0.0
            f[K] = 1.0
            for k in range(1, K):
                f[k] = float(_sigmoid(kappa[k - 1] - ei))
            for k in range(K):
                out[i, k] = max(f[k + 1] - f[k], 0.0)
            s = float(np.sum(out[i]))
            if s > 0:
                out[i] /= s
        return out


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
    ) -> None:
        if K < 2:
            raise BadInputError("K must be >= 2.")
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
        self.results: FederatedGLMMOrdinalResults | None = None

    @staticmethod
    def _add_intercept(X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.hstack([np.ones((X.shape[0], 1), dtype=float), X])

    @staticmethod
    def _site_derivatives(
        X: np.ndarray,
        y: np.ndarray,
        center_ids: np.ndarray,
        theta: np.ndarray,
        K: int,
    ) -> dict[str, Any]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        center_ids = np.asarray(center_ids)
        if X.ndim != 2:
            raise BadInputError(f"X must be 2D, got {X.shape}.")
        if y.shape[0] != X.shape[0] or center_ids.shape[0] != X.shape[0]:
            raise BadInputError("X, y, center_ids must have same length.")
        p = X.shape[1]
        kappas, log_su2 = _build_kappas_from_theta(theta, p, K)
        sigma_u2 = float(np.exp(log_su2))
        q = p + 1 + (K - 1)

        score = np.zeros(q, dtype=float)
        h = np.zeros((q, q), dtype=float)
        beta = np.asarray(theta[:p], dtype=float)
        gammas = np.asarray(theta[p + 2 : p + 1 + (K - 1)], dtype=float)
        exp_g = np.exp(gammas)

        uniq = np.unique(center_ids)
        sum_log_huu = 0.0
        for cid in uniq:
            mask = center_ids == cid
            xj = X[mask]
            yj = y[mask]
            if xj.size == 0:
                continue
            eta0 = xj @ beta
            u_star, huu = _mode_u_for_center_with_k(eta0, yj, kappas, sigma_u2)
            sum_log_huu += np.log(huu)

            grad_beta_sum = np.zeros(p, dtype=float)
            grad_kappa_sum = np.zeros(K - 1, dtype=float)
            for i in range(yj.size):
                eta = float(eta0[i] + u_star)
                _, dlog_deta, _, dlog_dk = _ordinal_pi_and_grads(
                    eta, int(yj[i]), kappas
                )
                dlog_dk = np.atleast_1d(dlog_dk)
                grad_beta_sum += dlog_deta * xj[i]
                grad_kappa_sum += dlog_dk

            grad_kappa1 = grad_kappa_sum[0] if K > 1 else 0.0
            grad_gammas = np.zeros_like(gammas)
            for g in range(gammas.size):
                grad_gammas[g] = exp_g[g] * np.sum(grad_kappa_sum[(g + 1) :])

            score[p] += 0.5 * (u_star * u_star / sigma_u2 - 1.0)
            score[:p] += grad_beta_sum
            if K > 1:
                score[p + 1] += grad_kappa1
            if K > 2:
                score[p + 2 : p + 1 + (K - 1)] += grad_gammas

            for i in range(yj.size):
                eta = float(eta0[i] + u_star)
                _, dlog_deta, _, dlog_dk = _ordinal_pi_and_grads(
                    eta, int(yj[i]), kappas
                )
                dlog_dk = np.atleast_1d(dlog_dk)
                g_beta_i = dlog_deta * xj[i]
                g_kappa1_i = dlog_dk[0] if K > 1 else 0.0
                g_gammas_i = np.zeros_like(gammas)
                for g in range(gammas.size):
                    g_gammas_i[g] = exp_g[g] * np.sum(dlog_dk[(g + 1) :])

                g_i = np.concatenate(
                    [
                        g_beta_i,
                        [0.0],
                        [g_kappa1_i] if K > 1 else [],
                        g_gammas_i,
                    ]
                )
                h -= np.outer(g_i, g_i)
            h[p, p] += -0.5

        h = 0.5 * (h + h.T)
        return {
            "q": int(q),
            "score": score.astype(float),
            "h_packed": np.asarray(pack_upper_triangle(h.astype(float)), dtype=float),
            "n_centers": int(len(uniq)),
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
        beta0 = np.zeros(p, dtype=float)
        kappa1_0 = 0.0
        gammas0 = np.zeros(max(0, self.K - 2), dtype=float)
        theta = np.concatenate(
            [
                beta0,
                np.array([self.log_sigma_u2_init], dtype=float),
                np.array([kappa1_0], dtype=float),
                gammas0,
            ]
        )

        history: list[dict[str, float]] = []
        converged = False
        for it in range(1, self.max_iters + 1):
            site = self._site_derivatives(X, y, center_ids, theta, self.K)
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
