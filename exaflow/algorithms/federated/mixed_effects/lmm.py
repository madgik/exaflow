from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from exaflow.algorithms.federated.mixed_effects.common import accumulate_gls_summaries
from exaflow.algorithms.federated.mixed_effects.common import (
    accumulate_reml_score_terms,
)
from exaflow.algorithms.federated.mixed_effects.common import apply_weights
from exaflow.algorithms.federated.mixed_effects.common import build_local_hist
from exaflow.algorithms.federated.mixed_effects.common import cluster_design_outcome
from exaflow.algorithms.federated.mixed_effects.common import compute_cluster_residuals
from exaflow.algorithms.federated.mixed_effects.common import (
    compute_vinv_random_intercept,
)
from exaflow.algorithms.federated.mixed_effects.common import extract_clusters
from exaflow.algorithms.federated.mixed_effects.common import gls_beta_from_sxx_sxy
from exaflow.algorithms.federated.mixed_effects.common import pack_upper_triangle
from exaflow.algorithms.federated.mixed_effects.common import (
    reml_objective_from_summaries,
)
from exaflow.algorithms.federated.mixed_effects.common import unpack_upper_triangle
from exaflow.algorithms.federated.mixed_effects.common import validate_inputs
from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.utils.agg_client import AggregationClient
from exaflow.algorithms.federated.utils.interfaces import FederatedEstimator
from exaflow.algorithms.federated.utils.interfaces import FederatedEstimatorResults

ALPHA = 0.05


class FederatedLMMResults(FederatedEstimatorResults):
    nobs: int

    def __init__(
        self,
        *,
        params: np.ndarray,
        bse: np.ndarray,
        tvalues: np.ndarray,
        pvalues: np.ndarray,
        conf_int_low: np.ndarray,
        conf_int_high: np.ndarray,
        cov_params: np.ndarray,
        sigma2: float,
        sigma_u2: float,
        nobs: int,
        n_groups: int,
        df_model: int,
        df_resid: int,
        ll_reml: float,
        aic: float,
        bic: float,
        converged: bool,
        n_iter: int,
        fit_intercept: bool,
        history: list[dict[str, float]] | None = None,
    ) -> None:
        self.params = np.asarray(params, dtype=float)
        self.bse = np.asarray(bse, dtype=float)
        self.tvalues = np.asarray(tvalues, dtype=float)
        self.pvalues = np.asarray(pvalues, dtype=float)
        self.conf_int_low = np.asarray(conf_int_low, dtype=float)
        self.conf_int_high = np.asarray(conf_int_high, dtype=float)
        self.cov_params = np.asarray(cov_params, dtype=float)
        self.sigma2 = float(sigma2)
        self.sigma_u2 = float(sigma_u2)
        self.nobs = int(nobs)
        self.n_groups = int(n_groups)
        self.df_model = int(df_model)
        self.df_resid = int(df_resid)
        self.ll_reml = float(ll_reml)
        self.aic = float(aic)
        self.bic = float(bic)
        self.converged = bool(converged)
        self.n_iter = int(n_iter)
        self.fit_intercept = bool(fit_intercept)
        self.history = history

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.fit_intercept:
            X = FederatedLMM._add_intercept(X)
        return X @ self.params


class _LMMRemlAggregator:
    def __init__(self, p: int, n_obs: int | None = None):
        self.p = int(p)
        self.n_obs = n_obs
        self.reset()

    def reset(self) -> None:
        self.sxx = np.zeros((self.p, self.p), dtype=float)
        self.sxy = np.zeros(self.p, dtype=float)
        self.syy = 0.0

        self.q1 = 0.0
        self.q2 = 0.0
        self.t0 = 0.0
        self.t1 = 0.0
        self.b = np.zeros((self.p, self.p), dtype=float)

        self.global_hist = np.zeros(0, dtype=np.int64)

    def accumulate(self, payload: dict[str, Any]) -> None:
        sxx_packed = payload.get("sxx_packed")
        if sxx_packed is not None:
            self.sxx += unpack_upper_triangle(sxx_packed, self.p)

        sxy = payload.get("sxy")
        if sxy is not None:
            self.sxy += np.asarray(sxy, dtype=float)

        syy = payload.get("syy")
        if syy is not None:
            self.syy += float(syy)

        q1 = payload.get("q1")
        if q1 is not None:
            self.q1 += float(q1)
        q2 = payload.get("q2")
        if q2 is not None:
            self.q2 += float(q2)
        t0 = payload.get("t0")
        if t0 is not None:
            self.t0 += float(t0)
        t1 = payload.get("t1")
        if t1 is not None:
            self.t1 += float(t1)

        b_packed = payload.get("b_packed")
        if b_packed is not None:
            self.b += unpack_upper_triangle(b_packed, self.p)

        gh = payload.get("global_hist")
        if gh is not None:
            self.global_hist = np.asarray(gh, dtype=np.int64)

    def compute_beta_gls(self, ridge: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
        return gls_beta_from_sxx_sxy(self.sxx, self.sxy, ridge=ridge)

    def objective(self, beta: np.ndarray, sigma2: float, sigma_u2: float) -> float:
        return reml_objective_from_summaries(
            self.sxx,
            self.syy,
            beta,
            sigma2=sigma2,
            sigma_u2=sigma_u2,
            hist=self.global_hist,
            n_obs=self.n_obs,
            p=self.p,
        )

    def gradient_logscale(
        self,
        beta: np.ndarray,
        sigma2: float,
        sigma_u2: float,
        *,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """
        Finite-difference gradient wrt (log sigma2, log sigma_u2) from the same
        objective used in line search. This guarantees objective/gradient consistency.
        """
        phi = np.array([np.log(sigma2), np.log(sigma_u2)], dtype=float)
        grad = np.zeros(2, dtype=float)
        for i in range(2):
            d = np.zeros(2, dtype=float)
            d[i] = eps
            sp = np.exp(phi + d)
            sm = np.exp(phi - d)
            fp = self.objective(beta, float(sp[0]), float(sp[1]))
            fm = self.objective(beta, float(sm[0]), float(sm[1]))
            grad[i] = (fp - fm) / (2.0 * eps)
        return grad


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

    def _rough_variance_init(
        self,
        y: np.ndarray,
        center_ids: np.ndarray,
        *,
        agg_client: AggregationClient,
    ) -> tuple[float, float]:
        # Global first/second moments of y
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

        # Federated per-center count/sum alignment via global union
        centers = list(agg_client.union(np.unique(center_ids).tolist()))
        if not centers:
            return max(total_var, self.lower_bound), self.lower_bound
        idx_map = {cid: i for i, cid in enumerate(centers)}
        m = len(centers)
        local_counts = np.zeros(m, dtype=float)
        local_sums = np.zeros(m, dtype=float)
        for cid in np.unique(center_ids):
            mask = center_ids == cid
            i = idx_map[cid]
            local_counts[i] = float(np.sum(mask))
            local_sums[i] = float(np.sum(y[mask]))
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
        within = total_var - between
        within = min(max(within, self.lower_bound), self.upper_bound)
        return within, between

    @staticmethod
    def _add_intercept(X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n = X.shape[0]
        return np.hstack([np.ones((n, 1), dtype=float), X])

    @staticmethod
    def _summaries_node(
        X: np.ndarray,
        y: np.ndarray,
        center_ids: np.ndarray,
        sigma2: float,
        sigma_u2: float,
        beta_hat: np.ndarray,
        w: np.ndarray | None = None,
        *,
        compute_reml_terms: bool = True,
    ) -> dict[str, Any]:
        validate_inputs(X, y, center_ids, w)
        Xw, yw = apply_weights(X, y, w)

        p = X.shape[1]
        sxx = np.zeros((p, p), dtype=float)
        sxy = np.zeros(p, dtype=float)
        syy = 0.0
        q1 = 0.0
        q2 = 0.0
        t0 = 0.0
        t1 = 0.0
        b = np.zeros((p, p), dtype=float)

        nj_sizes: list[int] = []
        clusters = extract_clusters(center_ids)
        for cid in clusters:
            xj, yj = cluster_design_outcome(Xw, yw, center_ids, cid)
            nj = int(xj.shape[0])
            if nj == 0:
                continue
            nj_sizes.append(nj)
            vj_inv, _ = compute_vinv_random_intercept(nj, sigma2, sigma_u2)
            sxx, sxy, syy = accumulate_gls_summaries(sxx, sxy, syy, xj, yj, vj_inv)
            if compute_reml_terms:
                rj = compute_cluster_residuals(xj, yj, beta_hat)
                q1, q2, t0, t1, b = accumulate_reml_score_terms(
                    q1, q2, t0, t1, b, xj, vj_inv, rj
                )

        payload: dict[str, Any] = {
            "p": p,
            "sxx_packed": pack_upper_triangle(sxx),
            "sxy": sxy.tolist(),
            "syy": float(syy),
            "nj_sizes": nj_sizes,
        }
        if compute_reml_terms:
            payload.update(
                {
                    "q1": float(q1),
                    "q2": float(q2),
                    "t0": float(t0),
                    "t1": float(t1),
                    "b_packed": pack_upper_triangle(b),
                }
            )
        return payload

    @staticmethod
    def _collect_global_phase_a(
        payload: dict[str, Any],
        agg_client: AggregationClient,
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        local_max, _ = build_local_hist(payload.get("nj_sizes", []))
        global_max = int(
            np.asarray(
                agg_client.max(np.array([float(local_max)], dtype=float)),
                dtype=float,
            ).reshape(-1)[0]
        )
        if global_max > 0:
            _, local_hist = build_local_hist(payload.get("nj_sizes", []), K=global_max)
            global_hist = np.asarray(agg_client.sum(local_hist.astype(float))).astype(
                np.int64
            )
        else:
            global_hist = np.zeros(0, dtype=np.int64)

        sxx = np.asarray(
            agg_client.sum(np.asarray(payload["sxx_packed"], dtype=float)),
            dtype=float,
        )
        sxy = np.asarray(
            agg_client.sum(np.asarray(payload["sxy"], dtype=float)),
            dtype=float,
        )
        syy = float(
            np.asarray(
                agg_client.sum(np.array([payload["syy"]], dtype=float)),
                dtype=float,
            ).reshape(-1)[0]
        )
        return sxx, sxy, syy, global_hist

    @staticmethod
    def _collect_global_phase_b(
        payload: dict[str, Any],
        agg_client: AggregationClient,
    ) -> tuple[float, float, float, float, np.ndarray]:
        q = np.asarray([payload["q1"], payload["q2"], payload["t0"], payload["t1"]])
        q_sum = np.asarray(agg_client.sum(q), dtype=float).reshape(-1)
        b_sum = np.asarray(
            agg_client.sum(np.asarray(payload["b_packed"], dtype=float)),
            dtype=float,
        )
        return float(q_sum[0]), float(q_sum[1]), float(q_sum[2]), float(q_sum[3]), b_sum

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

        p = X.shape[1]
        beta = np.zeros(p, dtype=float)
        if self.use_rough_init:
            sigma2, sigma_u2 = self._rough_variance_init(
                y,
                center_ids,
                agg_client=agg_client,
            )
        else:
            sigma2 = float(self.init_sigma2)
            sigma_u2 = float(self.init_sigma_u2)
        converged = False
        history: list[dict[str, float]] = []
        final_agg = _LMMRemlAggregator(p, n_obs=n_obs)
        final_a_inv = np.eye(p, dtype=float)

        for it in range(1, self.max_iter + 1):

            def _profile_eval(
                s2_eval: float,
                su2_eval: float,
            ) -> tuple[np.ndarray, float, _LMMRemlAggregator, np.ndarray]:
                phase_a_payload = self._summaries_node(
                    X,
                    y,
                    center_ids,
                    s2_eval,
                    su2_eval,
                    beta,
                    w=w,
                    compute_reml_terms=False,
                )
                sxx_packed, sxy, syy, global_hist = self._collect_global_phase_a(
                    phase_a_payload, agg_client
                )
                agg_eval = _LMMRemlAggregator(p, n_obs=n_obs)
                agg_eval.accumulate(
                    {
                        "sxx_packed": sxx_packed,
                        "sxy": sxy,
                        "syy": syy,
                        "global_hist": global_hist,
                    }
                )
                beta_eval, a_inv_eval = agg_eval.compute_beta_gls(ridge=self.ridge)
                ll_eval = agg_eval.objective(beta_eval, s2_eval, su2_eval)
                return beta_eval, float(ll_eval), agg_eval, a_inv_eval

            beta_new, ell0, agg, a_inv = _profile_eval(sigma2, sigma_u2)
            phi0 = np.array([np.log(sigma2), np.log(sigma_u2)], dtype=float)
            eps = 1e-6
            g = np.zeros(2, dtype=float)
            for i in range(2):
                d = np.zeros(2, dtype=float)
                d[i] = eps
                sp = np.exp(phi0 + d)
                sm = np.exp(phi0 - d)
                ll_p = _profile_eval(float(sp[0]), float(sp[1]))[1]
                ll_m = _profile_eval(float(sm[0]), float(sm[1]))[1]
                g[i] = (ll_p - ll_m) / (2.0 * eps)
            # Trust-region style regularization in step space
            d = g / (1.0 + self.reg_lambda)
            gd = float(g @ d)
            log_lo = float(np.log(self.lower_bound))
            log_hi = float(np.log(self.upper_bound))
            sigma2_new, sigma_u2_new = sigma2, sigma_u2
            beta_selected = beta_new
            agg_selected = agg
            a_inv_selected = a_inv
            improved = False

            if np.isfinite(gd) and gd > 0:
                step = 1.0
                ell_selected = ell0
                for _ in range(10):
                    phi_try = np.clip(phi0 + step * d, log_lo, log_hi)
                    s2_try = float(np.exp(phi_try[0]))
                    su2_try = float(np.exp(phi_try[1]))
                    beta_t, ell_try, agg_t, a_inv_t = _profile_eval(s2_try, su2_try)

                    if np.isfinite(ell_try) and ell_try >= ell0 + 1e-4 * step * gd:
                        sigma2_new, sigma_u2_new = s2_try, su2_try
                        beta_selected = beta_t
                        agg_selected = agg_t
                        a_inv_selected = a_inv_t
                        ell_selected = ell_try
                        improved = True
                        break
                    step *= 0.5
            else:
                ell_selected = ell0

            delta = max(
                float(
                    np.linalg.norm(beta_selected - beta) / (1.0 + np.linalg.norm(beta))
                ),
                abs(sigma2_new - sigma2) / (1.0 + sigma2),
                abs(sigma_u2_new - sigma_u2) / (1.0 + sigma_u2),
            )
            history.append(
                {
                    "iter": float(it),
                    "delta": float(delta),
                    "sigma2": float(sigma2_new),
                    "sigma_u2": float(sigma_u2_new),
                    "ll_reml": float(ell_selected),
                    "improved": float(improved),
                }
            )
            beta = beta_selected
            sigma2 = sigma2_new
            sigma_u2 = sigma_u2_new
            final_agg = agg_selected
            final_a_inv = a_inv_selected
            if it >= self.min_iter and delta < self.tol:
                converged = True
                break

        beta = np.asarray(beta, dtype=float)
        cov_params = final_a_inv * sigma2
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

        ll_reml = final_agg.objective(beta, sigma2, sigma_u2)
        n_groups = (
            int(np.sum(final_agg.global_hist)) if final_agg.global_hist.size else 0
        )
        k_params = p + 2
        aic = float(2.0 * k_params - 2.0 * ll_reml)
        bic = float(np.log(max(n_obs, 1)) * k_params - 2.0 * ll_reml)

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
            ll_reml=ll_reml,
            aic=aic,
            bic=bic,
            converged=converged,
            n_iter=len(history),
            fit_intercept=self.fit_intercept,
            history=history if self.return_history else None,
        )
        self.results = results
        return results
