# tests_new/datasets/synthetic_ordinal_glmm_dataset.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Tuple

from tests_new.utils.interfaces.partitioned_table import PartitionedPandasTable


def synth_glmm_ordinal_df(
    n_centers: int,
    n_features: int,
    n_min: int,
    n_max: int,
    K: int,
    beta: np.ndarray | None,
    cutpoints: np.ndarray | None,
    sigma_u2: float,
    seed: int,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, float, list[tuple[str, int]]]:
    """
    Synthetic ordinal GLMM data generator.
    cumulative logit: logit P(Y ≤ k | x, u_j) = κ_k - x^T β - u_j
    """
    rng = np.random.default_rng(seed)
    if beta is None:
        beta = np.array([-0.2] + [0.5] * n_features, dtype=float)
    if cutpoints is None:
        cutpoints = np.array([-1.0, 0.5, 1.6], dtype=float)[: K - 1]

    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    rows: list[pd.DataFrame] = []
    centers_info: list[tuple[str, int]] = []

    for j in range(n_centers):
        n_j = int(rng.integers(n_min, n_max + 1))
        X_cov = rng.normal(size=(n_j, n_features))
        X = np.hstack([np.ones((n_j, 1)), X_cov])

        u_j = rng.normal(loc=0.0, scale=np.sqrt(sigma_u2))
        eta = X @ beta + u_j

        cdfs = np.stack([_sigmoid(kappa - eta) for kappa in cutpoints], axis=1)
        probs = np.zeros((n_j, K))
        probs[:, 0] = cdfs[:, 0]
        for k in range(1, K - 1):
            probs[:, k] = cdfs[:, k] - cdfs[:, k - 1]
        probs[:, K - 1] = 1.0 - cdfs[:, -1]

        probs = np.clip(probs, 1e-9, 1.0)
        probs /= probs.sum(axis=1, keepdims=True)

        y = np.array([rng.choice(K, p=probs[i]) for i in range(n_j)], dtype=np.int64)
        c = np.full(n_j, f"C{j}")

        row: dict[str, Any] = {"center": c, "y": y}
        for kk in range(n_features):
            row[f"x{kk+1}"] = X_cov[:, kk]
        rows.append(pd.DataFrame(row))
        centers_info.append((f"C{j}", n_j))

    df = pd.concat(rows, ignore_index=True)
    return df, beta, cutpoints.astype(float), sigma_u2, centers_info


class SyntheticOrdinalGlmmDataset(PartitionedPandasTable):
    """
    MiniMip-style dataset wrapper for ordinal GLMM.
    Columns: center, y, x1, x2, ...
    """

    def __init__(
        self,
        n_centers: int = 12,
        n_features: int = 2,
        n_min: int = 50,
        n_max: int = 100,
        K: int = 4,
        sigma_u2: float = 0.5,
        seed: int = 2025,
    ):
        
        self.n_centers = n_centers
        self.n_features = n_features
        self.n_min = n_min
        self.n_max = n_max
        self.K = K
        self.sigma_u2 = sigma_u2
        self.seed = seed

        # ground truth
        self.true_beta: np.ndarray | None = None
        self.true_cutpoints: np.ndarray | None = None
        self.true_sigma_u2: float | None = None
        self.centers_info: list[tuple[str, int]] | None = None

        
        super().__init__()

    def get_dataset(self, *args, **kwargs) -> pd.DataFrame:  # type: ignore[override]
        df, beta, cutpoints, sigma_u2, centers_info = synth_glmm_ordinal_df(
            n_centers=self.n_centers,
            n_features=self.n_features,
            n_min=self.n_min,
            n_max=self.n_max,
            K=self.K,
            beta=None,
            cutpoints=None,
            sigma_u2=self.sigma_u2,
            seed=self.seed,
        )
        self.true_beta = beta
        self.true_cutpoints = cutpoints
        self.true_sigma_u2 = sigma_u2
        self.centers_info = centers_info
        return df
