from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from exaflow.algorithms.federated.mixed_effects.common import validate_inputs
from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.utils.agg_client import AggregationClient
from exaflow.algorithms.federated.utils.interfaces import FederatedEstimator
from exaflow.algorithms.federated.utils.interfaces import FederatedEstimatorResults

_SIG_CLIP = 40.0


def _sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    zc = np.clip(z, -_SIG_CLIP, _SIG_CLIP)
    return 1.0 / (1.0 + np.exp(-zc))


def _ordinal_param_dim(K: int, fit_intercept: bool) -> int:
    if K < 2:
        raise BadInputError("K must be >= 2.")
    if fit_intercept:
        return max(0, K - 2)
    return K - 1


def _build_kappas_from_theta(
    theta: np.ndarray,
    p: int,
    K: int,
    fit_intercept: bool,
) -> tuple[np.ndarray, float, np.ndarray]:
    theta = np.asarray(theta, dtype=float).reshape(-1)
    if K < 2:
        raise BadInputError(f"K must be >= 2 (got {K}).")
    ordinal_dim = _ordinal_param_dim(K, fit_intercept)
    expected_len = p + 1 + ordinal_dim
    if theta.shape[0] != expected_len:
        raise BadInputError(
            f"theta length {theta.shape[0]} != expected {expected_len} "
            f"(p={p}, K={K}, fit_intercept={fit_intercept})."
        )

    log_su2 = float(theta[p])
    kappas = np.empty(K - 1, dtype=float)
    if fit_intercept:
        gammas = np.asarray(theta[p + 1 :], dtype=float)
        kappas[0] = 0.0
        if K > 2:
            gaps = np.exp(gammas)
            for i in range(1, K - 1):
                kappas[i] = kappas[i - 1] + gaps[i - 1]
        return kappas, log_su2, gammas

    kappa1 = float(theta[p + 1])
    gammas = np.asarray(theta[p + 2 :], dtype=float)
    kappas[0] = kappa1
    if K > 2:
        gaps = np.exp(gammas)
        for i in range(1, K - 1):
            kappas[i] = kappas[i - 1] + gaps[i - 1]
    return kappas, log_su2, gammas


def _raw_cutpoint_gradient_to_theta(
    raw_grad: np.ndarray,
    exp_gammas: np.ndarray,
    fit_intercept: bool,
) -> np.ndarray:
    raw_grad = np.asarray(raw_grad, dtype=float).reshape(-1)
    exp_gammas = np.asarray(exp_gammas, dtype=float).reshape(-1)
    if fit_intercept:
        if raw_grad.size <= 1:
            return np.zeros(0, dtype=float)
        return exp_gammas * _suffix_sums(raw_grad[1:])

    theta_grad = np.zeros(raw_grad.size, dtype=float)
    if raw_grad.size == 0:
        return theta_grad
    theta_grad[0] = float(np.sum(raw_grad))
    if raw_grad.size > 1:
        theta_grad[1:] = exp_gammas * _suffix_sums(raw_grad[1:])
    return theta_grad


def _ordinal_loglik_derivatives_batch(
    eta: np.ndarray,
    y: np.ndarray,
    kappas: np.ndarray,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    eta = np.asarray(eta, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=int).reshape(-1)
    kappas = np.asarray(kappas, dtype=float).reshape(-1)
    K = kappas.shape[0] + 1
    if eta.shape[0] != y.shape[0]:
        raise BadInputError("eta and y must have same length.")
    if eta.size == 0:
        empty_k = np.zeros((0, K - 1), dtype=float)
        empty = np.zeros(0, dtype=float)
        return empty, empty, empty, empty, empty_k, empty_k, empty_k
    if np.any((y < 0) | (y > K - 1)):
        raise BadInputError(f"y must be in [0, {K - 1}].")

    a = kappas[None, :] - eta[:, None]
    s = np.asarray(_sigmoid(a), dtype=float)
    g1_inner = s * (1.0 - s)
    g2_inner = g1_inner * (1.0 - 2.0 * s)
    g3_inner = g1_inner * (1.0 - 6.0 * s + 6.0 * s * s)

    n = eta.size
    f = np.zeros((n, K + 1), dtype=float)
    g1 = np.zeros((n, K + 1), dtype=float)
    g2 = np.zeros((n, K + 1), dtype=float)
    g3 = np.zeros((n, K + 1), dtype=float)
    f[:, 1:K] = s
    f[:, K] = 1.0
    g1[:, 1:K] = g1_inner
    g2[:, 1:K] = g2_inner
    g3[:, 1:K] = g3_inner

    idx = np.arange(n)
    pi = np.clip(f[idx, y + 1] - f[idx, y], 1e-12, 1.0)
    pi_eta = -g1[idx, y + 1] + g1[idx, y]
    pi_eta2 = g2[idx, y + 1] - g2[idx, y]
    pi_eta3 = -g3[idx, y + 1] + g3[idx, y]

    dlog_deta = pi_eta / pi
    d2log_deta2 = pi_eta2 / pi - (pi_eta * pi_eta) / (pi * pi)
    d3log_deta3 = (
        pi_eta3 / pi
        - 3.0 * pi_eta2 * pi_eta / (pi * pi)
        + 2.0 * (pi_eta * pi_eta * pi_eta) / (pi * pi * pi)
    )

    dlog_dk = np.zeros((n, K - 1), dtype=float)
    d_deta_dk = np.zeros((n, K - 1), dtype=float)
    d_deta2_dk = np.zeros((n, K - 1), dtype=float)

    lower_mask = y >= 1
    if np.any(lower_mask):
        lower_idx = y[lower_mask] - 1
        row_idx = idx[lower_mask]
        pi_k = -g1[row_idx, y[lower_mask]]
        pi_eta_k = g2[row_idx, y[lower_mask]]
        pi_eta2_k = -g3[row_idx, y[lower_mask]]
        dlog_dk[row_idx, lower_idx] += pi_k / pi[lower_mask]
        d_deta_dk[row_idx, lower_idx] += pi_eta_k / pi[lower_mask] - (
            pi_eta[lower_mask] * pi_k
        ) / (pi[lower_mask] * pi[lower_mask])
        d_deta2_dk[row_idx, lower_idx] += (
            pi_eta2_k / pi[lower_mask]
            - (pi_eta2[lower_mask] * pi_k) / (pi[lower_mask] * pi[lower_mask])
            - 2.0 * dlog_deta[lower_mask] * d_deta_dk[row_idx, lower_idx]
        )

    upper_mask = y + 1 <= K - 1
    if np.any(upper_mask):
        upper_idx = y[upper_mask]
        row_idx = idx[upper_mask]
        pi_k = g1[row_idx, y[upper_mask] + 1]
        pi_eta_k = -g2[row_idx, y[upper_mask] + 1]
        pi_eta2_k = g3[row_idx, y[upper_mask] + 1]
        dlog_dk[row_idx, upper_idx] += pi_k / pi[upper_mask]
        d_deta_dk[row_idx, upper_idx] += pi_eta_k / pi[upper_mask] - (
            pi_eta[upper_mask] * pi_k
        ) / (pi[upper_mask] * pi[upper_mask])
        d_deta2_dk[row_idx, upper_idx] += (
            pi_eta2_k / pi[upper_mask]
            - (pi_eta2[upper_mask] * pi_k) / (pi[upper_mask] * pi[upper_mask])
            - 2.0 * dlog_deta[upper_mask] * d_deta_dk[row_idx, upper_idx]
        )

    return (
        np.log(pi),
        dlog_deta,
        d2log_deta2,
        d3log_deta3,
        dlog_dk,
        d_deta_dk,
        d_deta2_dk,
    )


@dataclass(frozen=True)
class _OrdinalParamState:
    theta: np.ndarray
    p: int
    K: int
    fit_intercept: bool
    beta: np.ndarray
    log_su2: float
    sigma_u2: float
    inv_su2: float
    kappas: np.ndarray
    gammas: np.ndarray
    exp_gammas: np.ndarray


def _build_param_state(
    theta: np.ndarray,
    *,
    p: int,
    K: int,
    fit_intercept: bool,
) -> _OrdinalParamState:
    theta = np.asarray(theta, dtype=float).reshape(-1)
    kappas, log_su2, gammas = _build_kappas_from_theta(theta, p, K, fit_intercept)
    sigma_u2 = float(np.exp(log_su2))
    return _OrdinalParamState(
        theta=theta,
        p=int(p),
        K=int(K),
        fit_intercept=bool(fit_intercept),
        beta=np.asarray(theta[:p], dtype=float),
        log_su2=float(log_su2),
        sigma_u2=sigma_u2,
        inv_su2=1.0 / sigma_u2,
        kappas=np.asarray(kappas, dtype=float),
        gammas=np.asarray(gammas, dtype=float),
        exp_gammas=np.exp(np.asarray(gammas, dtype=float)),
    )


def _converged(
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
        bse: np.ndarray,
        zvalues: np.ndarray,
        pvalues: np.ndarray,
        conf_int_low: np.ndarray,
        conf_int_high: np.ndarray,
        cov_params: np.ndarray,
        sigma_u2: float,
        cutpoints: np.ndarray,
        ll_laplace: float,
        aic: float,
        bic: float,
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
        self.bse = np.asarray(bse, dtype=float)
        self.zvalues = np.asarray(zvalues, dtype=float)
        self.pvalues = np.asarray(pvalues, dtype=float)
        self.conf_int_low = np.asarray(conf_int_low, dtype=float)
        self.conf_int_high = np.asarray(conf_int_high, dtype=float)
        self.cov_params = np.asarray(cov_params, dtype=float)
        self.sigma_u2 = float(sigma_u2)
        self.cutpoints = np.asarray(cutpoints, dtype=float)
        self.ll_laplace = float(ll_laplace)
        self.aic = float(aic)
        self.bic = float(bic)
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
        _, dlog_deta, d2log_deta2, _, _, _, _ = _ordinal_loglik_derivatives_batch(
            eta_base + u,
            y,
            kappas,
        )
        g_sum = -u / sigma_u2 + float(np.sum(dlog_deta))
        h_sum = -1.0 / sigma_u2 + float(np.sum(d2log_deta2))

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

    def _clip_theta_bounds(self, theta: np.ndarray, *, p: int) -> np.ndarray:
        theta = np.asarray(theta, dtype=float).copy()
        lo, hi = self.clip_log_sigma_bounds
        theta[p] = float(np.clip(theta[p], lo, hi))
        return theta

    def _cluster_objective_and_gradient(
        self,
        cluster: _OrdinalClusterData,
        param_state: _OrdinalParamState,
        *,
        init_u: float,
    ) -> tuple[float, np.ndarray, float]:
        p = param_state.p
        eta0 = cluster.X @ param_state.beta
        u_star, huu = _mode_u_for_center_with_k_warm(
            eta0,
            cluster.y,
            param_state.kappas,
            param_state.sigma_u2,
            init_u=float(init_u),
            max_iter=self.mode_max_iter,
            tol=self.mode_tol,
        )

        (
            logpi,
            dlog_deta,
            d2log_deta2,
            d3log_deta3,
            dlog_dk,
            d_deta_dk,
            d_deta2_dk,
        ) = _ordinal_loglik_derivatives_batch(
            eta0 + u_star,
            cluster.y,
            param_state.kappas,
        )
        sum_logpi = float(np.sum(logpi))
        sum_l3 = float(np.sum(d3log_deta3))
        grad_beta = cluster.X.T @ dlog_deta
        fu_beta = cluster.X.T @ d2log_deta2
        fuu_beta = cluster.X.T @ d3log_deta3
        grad_k_raw = np.sum(dlog_dk, axis=0)
        fu_k_raw = np.sum(d_deta_dk, axis=0)
        fuu_k_raw = np.sum(d_deta2_dk, axis=0)

        objective = (
            sum_logpi
            - 0.5 * param_state.log_su2
            - 0.5 * (u_star * u_star) * param_state.inv_su2
            - 0.5 * np.log(huu)
        )
        laplace_linear = 0.5 / huu
        laplace_cubic = 0.5 * sum_l3 / (huu * huu)

        grad_beta += laplace_linear * fuu_beta + laplace_cubic * fu_beta
        grad_phi = (
            0.5 * ((u_star * u_star) * param_state.inv_su2 - 1.0)
            + laplace_linear * param_state.inv_su2
            + laplace_cubic * (u_star * param_state.inv_su2)
        )
        grad_k_raw += laplace_linear * fuu_k_raw + laplace_cubic * fu_k_raw
        grad_ord = _raw_cutpoint_gradient_to_theta(
            grad_k_raw,
            param_state.exp_gammas,
            param_state.fit_intercept,
        )
        grad = np.concatenate(
            [grad_beta, np.array([grad_phi], dtype=float), grad_ord.astype(float)]
        )
        return float(objective), grad.astype(float), float(u_star)

    def _site_objective_gradient_from_clusters(
        self,
        clusters: list[_OrdinalClusterData],
        theta: np.ndarray,
        *,
        p: int,
        mode_state: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        param_state = _build_param_state(
            theta,
            p=p,
            K=self.K,
            fit_intercept=self.fit_intercept,
        )
        objective = 0.0
        grad = np.zeros_like(theta, dtype=float)
        next_mode_state = np.zeros_like(mode_state, dtype=float)
        for idx, cluster in enumerate(clusters):
            cluster_obj, cluster_grad, u_star = self._cluster_objective_and_gradient(
                cluster,
                param_state,
                init_u=float(mode_state[idx]),
            )
            objective += cluster_obj
            grad += cluster_grad
            next_mode_state[idx] = u_star
        return float(objective), grad.astype(float), next_mode_state

    def _site_objective_only_from_clusters(
        self,
        clusters: list[_OrdinalClusterData],
        theta: np.ndarray,
        *,
        p: int,
        mode_state: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        param_state = _build_param_state(
            theta,
            p=p,
            K=self.K,
            fit_intercept=self.fit_intercept,
        )
        objective = 0.0
        next_mode_state = np.zeros_like(mode_state, dtype=float)
        for idx, cluster in enumerate(clusters):
            cluster_obj, _, u_star = self._cluster_objective_and_gradient(
                cluster,
                param_state,
                init_u=float(mode_state[idx]),
            )
            objective += cluster_obj
            next_mode_state[idx] = u_star
        return float(objective), next_mode_state

    def _aggregate_objective_gradient(
        self,
        clusters: list[_OrdinalClusterData],
        theta: np.ndarray,
        *,
        p: int,
        mode_state: np.ndarray,
        agg_client: AggregationClient,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        objective_local, score_local, next_mode_state = (
            self._site_objective_gradient_from_clusters(
                clusters,
                theta,
                p=p,
                mode_state=mode_state,
            )
        )
        fused = np.concatenate(
            [np.array([objective_local], dtype=float), score_local.astype(float)]
        )
        fused_sum = np.asarray(agg_client.sum(fused), dtype=float).reshape(-1)
        objective = float(fused_sum[0])
        score = fused_sum[1:]
        return objective, score.astype(float), next_mode_state

    def _aggregate_objective_only(
        self,
        clusters: list[_OrdinalClusterData],
        theta: np.ndarray,
        *,
        p: int,
        mode_state: np.ndarray,
        agg_client: AggregationClient,
    ) -> tuple[float, np.ndarray]:
        objective_local, next_mode_state = self._site_objective_only_from_clusters(
            clusters,
            theta,
            p=p,
            mode_state=mode_state,
        )
        objective = float(
            np.asarray(
                agg_client.sum(np.array([objective_local], dtype=float)),
                dtype=float,
            ).reshape(-1)[0]
        )
        return objective, next_mode_state

    def _observed_information_finite_difference(
        self,
        clusters: list[_OrdinalClusterData],
        theta: np.ndarray,
        *,
        p: int,
        mode_state: np.ndarray,
        agg_client: AggregationClient,
        eps: float = 1e-5,
    ) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        q = theta.shape[0]
        jac = np.zeros((q, q), dtype=float)
        for idx in range(q):
            step = eps * max(1.0, abs(float(theta[idx])))
            theta_plus = theta.copy()
            theta_minus = theta.copy()
            theta_plus[idx] += step
            theta_minus[idx] -= step
            theta_plus = self._clip_theta_bounds(theta_plus, p=p)
            theta_minus = self._clip_theta_bounds(theta_minus, p=p)
            _, score_plus, _ = self._aggregate_objective_gradient(
                clusters,
                theta_plus,
                p=p,
                mode_state=mode_state,
                agg_client=agg_client,
            )
            _, score_minus, _ = self._aggregate_objective_gradient(
                clusters,
                theta_minus,
                p=p,
                mode_state=mode_state,
                agg_client=agg_client,
            )
            jac[:, idx] = (score_plus - score_minus) / (2.0 * step)
        observed_info = -0.5 * (jac + jac.T)
        return observed_info

    @staticmethod
    def _wald_inference_from_observed_info(
        observed_info: np.ndarray,
        params: np.ndarray,
        *,
        alpha: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        observed_info = 0.5 * (observed_info + observed_info.T)
        eig_min = float(np.min(np.linalg.eigvalsh(observed_info)))
        if eig_min <= 1e-10:
            observed_info = observed_info + (1e-10 - eig_min) * np.eye(
                observed_info.shape[0],
                dtype=float,
            )
        cov_theta = np.linalg.pinv(observed_info)
        p = params.shape[0]
        cov_params = cov_theta[:p, :p]
        bse = np.sqrt(np.maximum(np.diag(cov_params), 0.0))
        zvalues = np.divide(
            params,
            bse,
            out=np.full_like(params, np.nan, dtype=float),
            where=bse > 0.0,
        )
        pvalues = 2.0 * stats.norm.sf(np.abs(zvalues))
        zcrit = stats.norm.ppf(1.0 - alpha / 2.0)
        conf_low = params - zcrit * bse
        conf_high = params + zcrit * bse
        return cov_params, bse, zvalues, pvalues, conf_low, conf_high

    def _backtracking_ascent(
        self,
        clusters: list[_OrdinalClusterData],
        theta: np.ndarray,
        objective: float,
        score: np.ndarray,
        direction: np.ndarray,
        *,
        p: int,
        mode_state: np.ndarray,
        agg_client: AggregationClient,
        armijo_c: float = 1e-4,
        max_backtracks: int = 8,
        min_step: float = 1e-6,
    ) -> tuple[np.ndarray, float, np.ndarray, bool]:
        theta = np.asarray(theta, dtype=float)
        score = np.asarray(score, dtype=float)
        direction = np.asarray(direction, dtype=float)
        gtd = float(score @ direction)
        if not np.isfinite(gtd) or gtd <= 0.0:
            direction = score.copy()
            gtd = float(score @ direction)

        alpha = 1.0
        for _ in range(max_backtracks):
            if alpha < min_step:
                break
            theta_trial = self._clip_theta_bounds(theta + alpha * direction, p=p)
            objective_trial, mode_state_trial = self._aggregate_objective_only(
                clusters,
                theta_trial,
                p=p,
                mode_state=mode_state,
                agg_client=agg_client,
            )
            if objective_trial >= objective + armijo_c * alpha * gtd:
                return theta_trial, objective_trial, mode_state_trial, True
            alpha *= 0.5
        return theta, objective, mode_state, False

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
        validate_inputs(X, y, center_ids)
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
        ordinal_dim = _ordinal_param_dim(self.K, self.fit_intercept)
        q = p + 1 + ordinal_dim
        if self.fit_intercept:
            theta = np.concatenate(
                [
                    np.zeros(p, dtype=float),
                    np.array([self.log_sigma_u2_init], dtype=float),
                    np.zeros(ordinal_dim, dtype=float),
                ]
            )
        else:
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
        theta = self._clip_theta_bounds(theta, p=p)
        objective, score, mode_state = self._aggregate_objective_gradient(
            clusters,
            theta,
            p=p,
            mode_state=mode_state,
            agg_client=agg_client,
        )
        h_inv = np.eye(q, dtype=float)
        for it in range(1, self.max_iters + 1):
            direction = h_inv @ score
            step_norm = float(np.linalg.norm(direction))
            if step_norm > self.max_step_norm and step_norm > 0.0:
                direction *= self.max_step_norm / step_norm

            theta_new, objective_new, mode_state_new, accepted = (
                self._backtracking_ascent(
                    clusters,
                    theta,
                    objective,
                    score,
                    direction,
                    p=p,
                    mode_state=mode_state,
                    agg_client=agg_client,
                )
            )
            if not accepted:
                break

            objective_new, score_new, mode_state_new = (
                self._aggregate_objective_gradient(
                    clusters,
                    theta_new,
                    p=p,
                    mode_state=mode_state_new,
                    agg_client=agg_client,
                )
            )

            score_norm = float(np.linalg.norm(score_new))
            dtheta_max = float(np.max(np.abs(theta_new - theta)))
            history.append(
                {
                    "iter": float(it),
                    "score_norm": score_norm,
                    "dtheta_max": dtheta_max,
                }
            )

            if _converged(
                theta,
                theta_new,
                score_new,
                tol_theta=self.tol_theta,
                tol_score=self.tol_score,
            ):
                theta = theta_new
                objective = objective_new
                score = score_new
                mode_state = mode_state_new
                converged = True
                break

            s_vec = theta_new - theta
            y_vec = score - score_new
            sy = float(s_vec @ y_vec)
            if sy > 1e-10:
                rho = 1.0 / sy
                eye = np.eye(q, dtype=float)
                v_left = eye - rho * np.outer(s_vec, y_vec)
                v_right = eye - rho * np.outer(y_vec, s_vec)
                h_inv = v_left @ h_inv @ v_right + rho * np.outer(s_vec, s_vec)
            else:
                h_inv = np.eye(q, dtype=float)

            theta = theta_new
            objective = objective_new
            score = score_new
            mode_state = mode_state_new

        kappas, log_su2, _ = _build_kappas_from_theta(
            theta,
            p,
            self.K,
            self.fit_intercept,
        )
        observed_info = self._observed_information_finite_difference(
            clusters,
            theta,
            p=p,
            mode_state=mode_state,
            agg_client=agg_client,
        )
        params = theta[:p]
        cov_params, bse, zvalues, pvalues, conf_low, conf_high = (
            self._wald_inference_from_observed_info(observed_info, params)
        )
        k_params = theta.shape[0]
        aic = float(2.0 * k_params - 2.0 * objective)
        bic = float(np.log(max(n_obs, 1)) * k_params - 2.0 * objective)
        results = FederatedGLMMOrdinalResults(
            theta=theta,
            params=params,
            bse=bse,
            zvalues=zvalues,
            pvalues=pvalues,
            conf_int_low=conf_low,
            conf_int_high=conf_high,
            cov_params=cov_params,
            sigma_u2=float(np.exp(log_su2)),
            cutpoints=kappas,
            ll_laplace=float(objective),
            aic=aic,
            bic=bic,
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
