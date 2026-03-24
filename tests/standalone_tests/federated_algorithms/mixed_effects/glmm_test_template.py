from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GLMMCase:
    name: str
    beta: tuple[float, ...]
    sigma_u2: float
    seed: int
    fit_intercept: bool = True
    n_features: int = 2
    n_centers: int = 12
    n_min: int = 40
    n_max: int = 80
    cluster_sizes: tuple[int, ...] | None = None
    rho: float = 0.0
    K: int = 4
    cutpoints: tuple[float, ...] | None = None


def build_glmm_case_matrix() -> list[GLMMCase]:
    return [
        GLMMCase(
            name="balanced_reference",
            beta=(-0.4, 0.8, -0.7),
            sigma_u2=0.7,
            seed=101,
            n_centers=18,
            n_min=55,
            n_max=75,
            rho=0.2,
            K=4,
        ),
        GLMMCase(
            name="few_centers_large_n",
            beta=(-0.3, 0.75, -0.55),
            sigma_u2=0.8,
            seed=103,
            cluster_sizes=(140, 160, 180, 150, 170),
            rho=0.15,
            K=4,
        ),
        GLMMCase(
            name="many_centers_small_n",
            beta=(-0.25, 0.65, -0.45),
            sigma_u2=0.5,
            seed=107,
            n_centers=30,
            n_min=12,
            n_max=18,
            rho=0.1,
            K=5,
        ),
        GLMMCase(
            name="highly_unbalanced_clusters",
            beta=(-0.35, 0.9, -0.7),
            sigma_u2=0.7,
            seed=109,
            cluster_sizes=(8, 10, 12, 15, 18, 25, 35, 60, 90, 140, 210, 260),
            rho=0.25,
            K=4,
        ),
        GLMMCase(
            name="low_random_effect_boundary",
            beta=(-0.45, 0.8, -0.5),
            sigma_u2=0.05,
            seed=113,
            n_centers=20,
            n_min=80,
            n_max=110,
            rho=0.1,
            K=4,
        ),
        GLMMCase(
            name="high_random_effect_icc",
            beta=(-0.2, 0.7, -0.45),
            sigma_u2=1.8,
            seed=127,
            n_centers=18,
            n_min=50,
            n_max=70,
            rho=0.2,
            K=4,
        ),
        GLMMCase(
            name="near_zero_signal",
            beta=(-0.1, 0.25, -0.2),
            sigma_u2=0.6,
            seed=131,
            n_centers=24,
            n_min=70,
            n_max=90,
            rho=0.15,
            K=5,
        ),
        GLMMCase(
            name="correlated_predictors",
            beta=(-0.3, 1.0, -0.95),
            sigma_u2=0.8,
            seed=137,
            n_centers=22,
            n_min=70,
            n_max=90,
            rho=0.85,
            K=4,
        ),
        GLMMCase(
            name="no_intercept_model",
            beta=(1.1, -0.8),
            sigma_u2=0.5,
            seed=139,
            fit_intercept=False,
            n_features=2,
            n_centers=16,
            n_min=60,
            n_max=85,
            rho=0.2,
            K=4,
        ),
        GLMMCase(
            name="distribution_stress",
            beta=(-2.4, 0.95, -0.65),
            sigma_u2=0.9,
            seed=149,
            n_centers=20,
            n_min=80,
            n_max=100,
            rho=0.2,
            K=4,
            cutpoints=(-2.2, -0.2, 1.6),
        ),
    ]


CASE_MATRIX = build_glmm_case_matrix()


def build_feature_covariance(n_features: int, rho: float) -> np.ndarray:
    idx = np.arange(n_features)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def split_indices_by_center(
    center_ids: np.ndarray,
    n_workers: int,
    *,
    seed: int = 123,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = np.unique(center_ids)
    rng.shuffle(centers)
    buckets = [centers[i::n_workers] for i in range(n_workers)]
    return [np.where(np.isin(center_ids, bucket))[0] for bucket in buckets]


def build_binary_model_kwargs(
    *,
    fit_intercept: bool,
    add_laplace_corrections: bool = True,
    return_history: bool = True,
) -> dict[str, float | int | bool]:
    return dict(
        fit_intercept=fit_intercept,
        max_iters=50,
        ridge=1e-6,
        tol_theta=1e-6,
        tol_score=1e-4,
        add_laplace_corrections=add_laplace_corrections,
        max_step_norm=5.0,
        return_history=return_history,
    )


def build_ordinal_model_kwargs(
    *,
    fit_intercept: bool,
    K: int,
    return_history: bool = True,
) -> dict[str, float | int | bool]:
    return dict(
        K=K,
        fit_intercept=fit_intercept,
        max_iters=50,
        ridge=1e-6,
        tol_theta=1e-6,
        tol_score=1e-4,
        max_step_norm=5.0,
        return_history=return_history,
    )


def synth_glmm_binary_case(case: GLMMCase) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(case.seed)
    beta = np.asarray(case.beta, dtype=float)
    expected_beta_len = case.n_features + (1 if case.fit_intercept else 0)
    if beta.shape[0] != expected_beta_len:
        raise ValueError(
            f"beta length {beta.shape[0]} does not match expected {expected_beta_len}."
        )

    if case.cluster_sizes is not None:
        cluster_sizes = np.asarray(case.cluster_sizes, dtype=int)
    else:
        cluster_sizes = rng.integers(case.n_min, case.n_max + 1, size=case.n_centers)

    cov = build_feature_covariance(case.n_features, case.rho)
    x_rows = []
    y_rows = []
    c_rows = []
    for j, nj in enumerate(cluster_sizes):
        x_cov = rng.multivariate_normal(
            mean=np.zeros(case.n_features),
            cov=cov,
            size=int(nj),
        )
        x_design = x_cov
        if case.fit_intercept:
            x_design = np.hstack([np.ones((int(nj), 1), dtype=float), x_cov])

        u_j = rng.normal(loc=0.0, scale=np.sqrt(case.sigma_u2))
        eta = x_design @ beta + u_j
        p = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(n=1, p=p, size=int(nj)).astype(float)
        x_rows.append(x_cov.astype(float))
        y_rows.append(y)
        c_rows.append(np.full(int(nj), f"C{j}", dtype=object))

    X = np.vstack(x_rows).astype(float)
    y = np.concatenate(y_rows).astype(float)
    center_ids = np.concatenate(c_rows)
    return X, y, center_ids


def synth_glmm_ordinal_case(case: GLMMCase) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(case.seed)
    beta = np.asarray(case.beta, dtype=float)
    expected_beta_len = case.n_features + (1 if case.fit_intercept else 0)
    if beta.shape[0] != expected_beta_len:
        raise ValueError(
            f"beta length {beta.shape[0]} does not match expected {expected_beta_len}."
        )

    if case.cluster_sizes is not None:
        cluster_sizes = np.asarray(case.cluster_sizes, dtype=int)
    else:
        cluster_sizes = rng.integers(case.n_min, case.n_max + 1, size=case.n_centers)

    if case.cutpoints is None:
        cutpoints = np.linspace(-1.0, 1.0, case.K - 1, dtype=float)
    else:
        cutpoints = np.asarray(case.cutpoints, dtype=float)
    if cutpoints.shape[0] != case.K - 1:
        raise ValueError("cutpoints length must be K-1")

    cov = build_feature_covariance(case.n_features, case.rho)
    x_rows = []
    y_rows = []
    c_rows = []
    for j, nj in enumerate(cluster_sizes):
        x_cov = rng.multivariate_normal(
            mean=np.zeros(case.n_features),
            cov=cov,
            size=int(nj),
        )
        x_design = x_cov
        if case.fit_intercept:
            x_design = np.hstack([np.ones((int(nj), 1), dtype=float), x_cov])

        u_j = rng.normal(loc=0.0, scale=np.sqrt(case.sigma_u2))
        eta = x_design @ beta + u_j
        y = np.zeros(int(nj), dtype=int)
        for i in range(int(nj)):
            f = np.zeros(case.K + 1, dtype=float)
            f[0] = 0.0
            f[case.K] = 1.0
            for k in range(1, case.K):
                f[k] = 1.0 / (1.0 + np.exp(-(cutpoints[k - 1] - eta[i])))
            probs = np.maximum(f[1:] - f[:-1], 0.0)
            probs = probs / np.sum(probs)
            y[i] = int(rng.choice(np.arange(case.K), p=probs))

        x_rows.append(x_cov.astype(float))
        y_rows.append(y)
        c_rows.append(np.full(int(nj), f"C{j}", dtype=object))

    X = np.vstack(x_rows).astype(float)
    y = np.concatenate(y_rows).astype(int)
    center_ids = np.concatenate(c_rows)
    return X, y, center_ids
