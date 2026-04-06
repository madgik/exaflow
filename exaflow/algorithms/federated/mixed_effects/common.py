from __future__ import annotations

from typing import Callable
from typing import Iterable

import numpy as np

from exaflow.algorithms.federated.utils import BadInputError


def validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    center_ids: np.ndarray,
    w: np.ndarray | None = None,
) -> None:
    if X.ndim != 2:
        raise BadInputError(f"X must be 2D, got shape {X.shape}.")
    n, _ = X.shape
    if y.ndim != 1 or y.shape[0] != n:
        raise BadInputError(f"y must be 1D with len=={n}, got shape {y.shape}.")
    if center_ids.ndim != 1 or center_ids.shape[0] != n:
        raise BadInputError(
            f"center_ids must be 1D with len=={n}, got shape {center_ids.shape}."
        )

    if not np.isfinite(X).all():
        raise BadInputError("X contains NaN/Inf.")
    if not np.isfinite(y).all():
        raise BadInputError("y contains NaN/Inf.")
    if w is not None:
        if w.ndim != 1 or w.shape[0] != n:
            raise BadInputError(f"w must be 1D with len=={n}, got shape {w.shape}.")
        if (w < 0).any() or not np.isfinite(w).all():
            raise BadInputError("w must be non-negative and finite.")


def apply_weights(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if w is None:
        return X, y
    s = np.sqrt(w).astype(float)
    return X * s[:, None], y * s


def extract_clusters(center_ids: np.ndarray) -> np.ndarray:
    return np.unique(center_ids)


def cluster_design_outcome(
    X: np.ndarray,
    y: np.ndarray,
    center_ids: np.ndarray,
    cluster_label,
) -> tuple[np.ndarray, np.ndarray]:
    idx = center_ids == cluster_label
    return X[idx], y[idx]


def compute_vinv_random_intercept(
    nj: int,
    sigma2: float,
    sigma_u2: float,
) -> tuple[np.ndarray, float]:
    if nj <= 0:
        raise BadInputError("nj must be positive.")
    if sigma2 <= 0 or sigma_u2 < 0:
        raise BadInputError("Require sigma2>0 and sigma_u2>=0.")

    inv_sigma2 = 1.0 / sigma2
    alpha_j = sigma_u2 / (sigma2 * (sigma2 + nj * sigma_u2))
    ones = np.ones((nj, 1), dtype=float)
    vj_inv = inv_sigma2 * np.eye(nj, dtype=float) - alpha_j * (ones @ ones.T)
    return vj_inv, alpha_j


def logdet_v_random_intercept(nj: int, sigma2: float, sigma_u2: float) -> float:
    if nj <= 0:
        raise BadInputError("nj must be positive.")
    if sigma2 <= 0 or sigma_u2 < 0:
        raise BadInputError("Require sigma2>0 and sigma_u2>=0.")
    return (nj - 1) * np.log(sigma2) + np.log(sigma2 + nj * sigma_u2)


def accumulate_gls_summaries(
    sxx: np.ndarray,
    sxy: np.ndarray,
    syy: float,
    xj: np.ndarray,
    yj: np.ndarray,
    vj_inv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    sxx += xj.T @ vj_inv @ xj
    sxy += xj.T @ vj_inv @ yj
    syy += float(yj.T @ vj_inv @ yj)
    return sxx, sxy, syy


def compute_cluster_residuals(
    xj: np.ndarray,
    yj: np.ndarray,
    beta_hat: np.ndarray,
) -> np.ndarray:
    return yj - xj @ beta_hat


def accumulate_reml_score_terms(
    q1: float,
    q2: float,
    t0: float,
    t1: float,
    b: np.ndarray,
    xj: np.ndarray,
    vj_inv: np.ndarray,
    rj: np.ndarray,
) -> tuple[float, float, float, float, np.ndarray]:
    nj = xj.shape[0]
    ones = np.ones((nj, 1), dtype=float)

    vjr = vj_inv @ rj
    q1 += float(vjr @ vjr)

    s1 = float(ones.T @ vjr)
    q2 += s1 * s1

    t0 += float(np.trace(vj_inv))
    t1 += float((ones.T @ vj_inv @ ones).squeeze())

    vj = xj.T @ (vj_inv @ ones)
    b += vj @ vj.T
    return q1, q2, t0, t1, b


def build_local_hist(
    nj_sizes: Iterable[int], K: int | None = None
) -> tuple[int, np.ndarray]:
    sizes = [int(x) for x in nj_sizes if int(x) > 0]
    if not sizes:
        return 0, np.zeros(0, dtype=np.int64)
    local_max = max(sizes)
    if K is None:
        K = local_max
    hist = np.zeros(K, dtype=np.int64)
    for n in sizes:
        if 1 <= n <= K:
            hist[n - 1] += 1
    return local_max, hist


def pack_upper_triangle(m: np.ndarray) -> list[float]:
    q = m.shape[0]
    iu = np.triu_indices(q)
    return m[iu].astype(np.float64).tolist()


def unpack_upper_triangle(v: list[float] | np.ndarray, p: int) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    m = np.zeros((p, p), dtype=np.float64)
    iu = np.triu_indices(p)
    m[iu] = arr
    m[(iu[1], iu[0])] = arr
    return m


def gls_beta_from_sxx_sxy(
    sxx: np.ndarray,
    sxy: np.ndarray,
    ridge: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    p = sxx.shape[0]
    a = sxx + ridge * np.eye(p, dtype=sxx.dtype)
    a_inv = np.linalg.pinv(a)
    beta = a_inv @ sxy
    return beta, a_inv


def sum_logdet_random_intercept_from_hist(
    hist: np.ndarray,
    sigma2: float,
    sigma_u2: float,
) -> float:
    if sigma2 <= 0 or sigma_u2 < 0:
        raise BadInputError("Require sigma2>0 and sigma_u2>=0.")
    if hist.size == 0:
        return 0.0
    k = np.arange(1, hist.size + 1, dtype=float)
    terms = (k - 1.0) * np.log(sigma2) + np.log(sigma2 + k * sigma_u2)
    return float(np.dot(hist.astype(float), terms))


def reml_objective_from_summaries(
    sxx: np.ndarray,
    syy: float,
    beta: np.ndarray,
    *,
    sigma2: float,
    sigma_u2: float,
    hist: np.ndarray,
    n_obs: int | None = None,
    p: int | None = None,
) -> float:
    logdet_v = sum_logdet_random_intercept_from_hist(hist, sigma2, sigma_u2)
    sign, logdet_sxx = np.linalg.slogdet(
        sxx + 1e-12 * np.eye(sxx.shape[0], dtype=sxx.dtype)
    )
    if sign <= 0:
        logdet_sxx = np.log(
            np.linalg.det(sxx + 1e-6 * np.eye(sxx.shape[0], dtype=sxx.dtype))
        )

    quad = float(beta @ (sxx @ beta))
    ell = -0.5 * (logdet_v + logdet_sxx + float(syy) - quad)
    # Add Gaussian constant term for direct comparability to statsmodels llf.
    if n_obs is not None and p is not None and n_obs > p:
        ell += -0.5 * float(n_obs - p) * float(np.log(2.0 * np.pi))
    return float(ell)


def reml_grad_logscale_from_summaries(
    q1: float,
    q2: float,
    t0: float,
    t1: float,
    b: np.ndarray,
    a_inv: np.ndarray,
    p: int,
    sigma2: float,
    sigma_u2: float,
) -> np.ndarray:
    tr_ainv_b = float(np.trace(a_inv @ b))
    u_s2 = 0.5 * (float(q1) - (float(t0) - float(p)))
    u_su2 = 0.5 * (float(q2) - (float(t1) - tr_ainv_b))
    g1 = float(sigma2) * u_s2
    g2 = float(sigma_u2) * u_su2
    return np.array([g1, g2], dtype=np.float64)


def backtracking_line_search_log2d(
    ell_fn: Callable[[np.ndarray], float],
    grad_fn: Callable[[np.ndarray], np.ndarray],
    phi0: np.ndarray,
    *,
    alpha: float = 1.0,
    c: float = 1e-4,
    max_backtracks: int = 8,
    min_step: float = 1e-6,
    lower_bound: float = 1e-12,
) -> tuple[np.ndarray, bool]:
    phi0 = np.asarray(phi0, dtype=float).reshape(2)
    ell0 = float(ell_fn(phi0))
    g = np.asarray(grad_fn(phi0), dtype=float).reshape(2)
    # REML objective is maximized; take an ascent direction.
    d = g

    if np.allclose(g, 0.0):
        return phi0, True

    step = float(alpha)
    for _ in range(max_backtracks):
        phi = phi0 + step * d
        s2, su2 = float(np.exp(phi[0])), float(np.exp(phi[1]))
        if (s2 < lower_bound) or (su2 < lower_bound):
            step *= 0.5
            if step < min_step:
                return phi0, False
            continue

        ell = float(ell_fn(phi))
        if ell >= ell0 + c * step * float(g @ d):
            return phi, True
        step *= 0.5
        if step < min_step:
            break

    return phi0, False


def clip_probs(p, eps: float = 1e-8):
    return np.clip(p, eps, 1.0 - eps)


def logistic_sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def glmm_binary_random_intercept_mode(
    eta_base: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    sigma_u2: float,
    max_iter: int = 25,
    tol: float = 1e-8,
) -> tuple[float, float]:
    if w is None:
        w = np.ones_like(y, dtype=float)
    inv_su2 = 1.0 / sigma_u2
    u = 0.0
    h = -inv_su2
    for _ in range(max_iter):
        eta = eta_base + u
        p = clip_probs(logistic_sigmoid(eta))
        g = np.sum(w * (y - p)) - u * inv_su2
        h = -np.sum(w * p * (1.0 - p)) - inv_su2
        step = g / h
        u_new = u - step
        if abs(u_new - u) < tol:
            u = u_new
            break
        u = u_new
    huu = -h
    return float(u), float(huu)


def glm_logistic_score_hessian_block(
    xj: np.ndarray,
    yj: np.ndarray,
    pj: np.ndarray,
    wj: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rj = (yj - pj) * wj
    vj = (pj * (1.0 - pj)) * wj
    s_beta = xj.T @ rj
    h_bb = -(xj.T * vj) @ xj
    return s_beta, h_bb


def glmm_laplace_corrections_beta(
    xj: np.ndarray,
    pj: np.ndarray,
    wj: np.ndarray,
    huu: float,
) -> np.ndarray:
    d_huu_deta = (pj * (1.0 - pj)) * (1.0 - 2.0 * pj) * wj
    return 0.5 * (xj.T @ d_huu_deta) / float(huu)
