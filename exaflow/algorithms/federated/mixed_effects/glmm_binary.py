from __future__ import annotations

from typing import Any

import numpy as np

from exaflow.algorithms.federated.mixed_effects.common import clip_probs
from exaflow.algorithms.federated.mixed_effects.common import (
    glm_logistic_score_hessian_block,
)
from exaflow.algorithms.federated.mixed_effects.common import (
    glmm_binary_random_intercept_mode,
)
from exaflow.algorithms.federated.mixed_effects.common import (
    glmm_laplace_corrections_beta,
)
from exaflow.algorithms.federated.mixed_effects.common import logistic_sigmoid
from exaflow.algorithms.federated.mixed_effects.common import pack_upper_triangle
from exaflow.algorithms.federated.mixed_effects.common import unpack_upper_triangle
from exaflow.algorithms.federated.mixed_effects.common import validate_inputs
from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.utils.agg_client import AggregationClient
from exaflow.algorithms.federated.utils.interfaces import FederatedEstimator
from exaflow.algorithms.federated.utils.interfaces import FederatedEstimatorResults


class FederatedGLMMBinaryResults(FederatedEstimatorResults):
    nobs: int

    def __init__(
        self,
        *,
        theta: np.ndarray,
        params: np.ndarray,
        sigma_u2: float,
        nobs: int,
        n_groups: int,
        converged: bool,
        n_iter: int,
        fit_intercept: bool,
        history: list[dict[str, float]] | None = None,
    ) -> None:
        self.theta = np.asarray(theta, dtype=float)
        self.params = np.asarray(params, dtype=float)
        self.sigma_u2 = float(sigma_u2)
        self.nobs = int(nobs)
        self.n_groups = int(n_groups)
        self.converged = bool(converged)
        self.n_iter = int(n_iter)
        self.fit_intercept = bool(fit_intercept)
        self.history = history

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.fit_intercept:
            X = FederatedGLMMBinary._add_intercept(X)
        eta = X @ self.params
        return logistic_sigmoid(eta)


class _GLMMAggregator:
    def __init__(self, q: int):
        self.q = int(q)
        self.reset()

    def reset(self) -> None:
        self.s = np.zeros(self.q, dtype=np.float64)
        self.h = np.zeros((self.q, self.q), dtype=np.float64)

    def accumulate(self, payload: dict[str, Any]) -> None:
        score = payload.get("score")
        h_packed = payload.get("h_packed")
        if score is None or h_packed is None:
            raise BadInputError("payload must contain 'score' and 'h_packed'")

        s = np.asarray(score, dtype=np.float64).reshape(-1)
        if s.shape[0] != self.q:
            raise BadInputError(f"score length {s.shape[0]} != q={self.q}")

        h_block = unpack_upper_triangle(np.asarray(h_packed, dtype=np.float64), self.q)
        h_block = 0.5 * (h_block + h_block.T)
        self.s += s
        self.h += h_block

    def newton_update(
        self,
        theta: np.ndarray,
        *,
        ridge: float,
        max_tries: int,
        max_step_norm: float,
        clip_log_sigma_bounds: tuple[float, float] | None,
    ) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float64).reshape(-1)
        if theta.shape[0] != self.q:
            raise BadInputError(f"theta length {theta.shape[0]} != q={self.q}")

        h = 0.5 * (self.h + self.h.T)
        s = self.s.copy()
        eye = np.eye(self.q, dtype=np.float64)
        lam = max(float(ridge), 0.0)
        delta = None

        for _ in range(int(max_tries)):
            try:
                delta = np.linalg.solve(h + lam * eye, s)
                break
            except np.linalg.LinAlgError:
                lam = max(lam * 10.0, 1e-12)

        if delta is None:
            delta = np.linalg.pinv(h + max(lam, 1e-6) * eye) @ s

        step_norm = float(np.linalg.norm(delta))
        if step_norm > max_step_norm and step_norm > 0.0:
            delta *= max_step_norm / step_norm

        theta_new = theta - delta
        if clip_log_sigma_bounds is not None:
            lo, hi = clip_log_sigma_bounds
            theta_new[-1] = float(np.clip(theta_new[-1], lo, hi))
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


class FederatedGLMMBinary(FederatedEstimator):
    def __init__(
        self,
        *,
        fit_intercept: bool = True,
        max_iters: int = 50,
        ridge: float = 1e-6,
        tol_theta: float = 1e-6,
        tol_score: float = 1e-4,
        add_laplace_corrections: bool = True,
        clip_log_sigma_bounds: tuple[float, float] = (np.log(1e-8), np.log(1e3)),
        max_step_norm: float = 5.0,
        log_sigma_u2_init: float = np.log(0.3),
        return_history: bool = False,
    ) -> None:
        if max_iters <= 0:
            raise BadInputError("max_iters must be positive.")
        if ridge < 0:
            raise BadInputError("ridge must be non-negative.")
        if tol_theta <= 0 or tol_score <= 0:
            raise BadInputError("tol_theta and tol_score must be positive.")
        if max_step_norm <= 0:
            raise BadInputError("max_step_norm must be positive.")

        self.fit_intercept = bool(fit_intercept)
        self.max_iters = int(max_iters)
        self.ridge = float(ridge)
        self.tol_theta = float(tol_theta)
        self.tol_score = float(tol_score)
        self.add_laplace_corrections = bool(add_laplace_corrections)
        self.clip_log_sigma_bounds = clip_log_sigma_bounds
        self.max_step_norm = float(max_step_norm)
        self.log_sigma_u2_init = float(log_sigma_u2_init)
        self.return_history = bool(return_history)
        self.results: FederatedGLMMBinaryResults | None = None

    @staticmethod
    def _add_intercept(X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n = X.shape[0]
        return np.hstack([np.ones((n, 1), dtype=float), X])

    @staticmethod
    def _site_derivatives(
        X: np.ndarray,
        y: np.ndarray,
        center_ids: np.ndarray,
        theta: np.ndarray,
        w: np.ndarray | None,
        add_laplace_corrections: bool,
    ) -> dict[str, Any]:
        validate_inputs(X, y, center_ids, w)
        n, p = X.shape
        theta = np.asarray(theta, dtype=float).reshape(-1)
        q = int(theta.shape[0])
        if q != p + 1:
            raise BadInputError(f"theta must have length p+1={p + 1}, got {q}")

        if w is None:
            w = np.ones_like(y, dtype=float)
        else:
            w = np.asarray(w, dtype=float)

        beta = theta[:p]
        log_su2 = float(theta[p])
        sigma_u2 = float(np.exp(log_su2))
        if sigma_u2 <= 0:
            raise BadInputError("sigma_u2 must be positive.")
        inv_su2 = 1.0 / sigma_u2
        eta0 = X @ beta

        score = np.zeros(q, dtype=float)
        h = np.zeros((q, q), dtype=float)
        uniq_centers = np.unique(center_ids)
        sum_log_huu = 0.0

        s_beta = score[:p]
        s_logsu2 = score[p : p + 1]
        h_bb = h[:p, :p]
        h_bs = h[:p, p : p + 1]
        h_ss = h[p : p + 1, p : p + 1]

        for j in uniq_centers:
            idx = center_ids == j
            xj = X[idx, :]
            yj = y[idx]
            wj = w[idx]
            eta_base = eta0[idx]

            u_star, huu = glmm_binary_random_intercept_mode(eta_base, yj, wj, sigma_u2)
            sum_log_huu += np.log(huu)

            eta = eta_base + u_star
            pj = clip_probs(logistic_sigmoid(eta))

            s_b, h_b = glm_logistic_score_hessian_block(xj, yj, pj, wj)
            s_beta += s_b
            h_bb += h_b

            if add_laplace_corrections:
                corr_beta = glmm_laplace_corrections_beta(xj, pj, wj, huu)
                s_beta += corr_beta
                h_bs += 0.5 * (corr_beta[:, None]) * (inv_su2 / (huu * huu))
                s_logsu2[0] += 0.5 * ((u_star * u_star) * inv_su2 - 1.0) - 0.5 * (
                    inv_su2 / huu
                )
                h_ss[0, 0] += -0.5 * (u_star * u_star) * inv_su2 - 0.5 * (inv_su2 / huu)

        h[p : p + 1, :p] = h_bs.T
        return {
            "q": q,
            "score": score.astype(float),
            "h_packed": np.asarray(pack_upper_triangle(h.astype(float)), dtype=float),
            "n_centers": int(len(uniq_centers)),
            "sum_log_huu": float(sum_log_huu),
        }

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        center_ids: np.ndarray,
        agg_client: AggregationClient,
        w: np.ndarray | None = None,
    ) -> FederatedGLMMBinaryResults:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        center_ids = np.asarray(center_ids)
        if self.fit_intercept:
            X = self._add_intercept(X)
        validate_inputs(X, y, center_ids, w)

        if not np.all(np.isin(y, [0.0, 1.0])):
            raise BadInputError("GLMM binary expects y in {0,1}.")

        n_obs = int(
            np.asarray(
                agg_client.sum(np.array([float(X.shape[0])], dtype=float)),
                dtype=float,
            ).reshape(-1)[0]
        )
        try:
            n_groups = len(agg_client.union(center_ids.tolist()))
        except Exception:
            n_groups = len(np.unique(center_ids))

        p = X.shape[1]
        theta = np.concatenate(
            [np.zeros(p, dtype=float), np.array([self.log_sigma_u2_init], dtype=float)]
        )
        q = p + 1
        history: list[dict[str, float]] = []
        converged = False

        for it in range(1, self.max_iters + 1):
            site = self._site_derivatives(
                X,
                y,
                center_ids,
                theta,
                w,
                self.add_laplace_corrections,
            )
            score_sum = np.asarray(agg_client.sum(site["score"]), dtype=float).reshape(
                q
            )
            h_packed_sum = np.asarray(
                agg_client.sum(np.asarray(site["h_packed"], dtype=float)),
                dtype=float,
            )

            agg = _GLMMAggregator(q)
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
                {
                    "iter": float(it),
                    "score_norm": score_norm,
                    "dtheta_max": dtheta_max,
                }
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

        params = theta[:p]
        sigma_u2 = float(np.exp(theta[p]))
        results = FederatedGLMMBinaryResults(
            theta=theta,
            params=params,
            sigma_u2=sigma_u2,
            nobs=n_obs,
            n_groups=n_groups,
            converged=converged,
            n_iter=len(history),
            fit_intercept=self.fit_intercept,
            history=history if self.return_history else None,
        )
        self.results = results
        return results
