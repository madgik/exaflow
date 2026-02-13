from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tests_new.utils.interfaces.partitioned_table import PartitionedPandasTable


class SyntheticGlmmDataset(PartitionedPandasTable):
    """
    Synthetic binary GLMM dataset for random-intercept logistic mixed models.

    Columns:
      - center: cluster label (C0, C1, ...)
      - y: binary outcome (0/1)
      - x1, x2, ...: covariates
    """

    def __init__(
        self,
        n_centers: int = 15,
        n_features: int = 2,
        n_min: int = 40,
        n_max: int = 80,
        beta: np.ndarray | None = None,
        sigma_u2: float = 0.6,
        seed: int = 42,
    ) -> None:
        # Store parameters FIRST so that get_dataset can access them
        self.n_centers = int(n_centers)
        self.n_features = int(n_features)
        self.n_min = int(n_min)
        self.n_max = int(n_max)
        self.sigma_u2 = float(sigma_u2)
        self.seed = int(seed)

        if beta is None:
            # Intercept + n_features slopes
            self.beta = np.array([-0.3] + [0.6] * self.n_features, dtype=float)
        else:
            self.beta = np.asarray(beta, dtype=float).reshape(-1)
            if self.beta.shape[0] != self.n_features + 1:
                raise ValueError(
                    f"beta must have length n_features+1={self.n_features+1}, "
                    f"got {self.beta.shape[0]}"
                )

        # PartitionedPandasTable will call self.get_dataset()
        super().__init__()

    def get_dataset(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Build a pandas DataFrame with (center, y, x1, x2, ...).
        """
        rng = np.random.default_rng(self.seed)
        rows: list[pd.DataFrame] = []

        for j in range(self.n_centers):
            n_j = int(rng.integers(self.n_min, self.n_max + 1))

            # Covariates
            X_cov = rng.normal(size=(n_j, self.n_features))
            intercept = np.ones((n_j, 1), dtype=float)
            X = np.hstack([intercept, X_cov])

            # Random intercept per center
            u_j = rng.normal(loc=0.0, scale=np.sqrt(self.sigma_u2))

            eta = X @ self.beta + u_j
            p = 1.0 / (1.0 + np.exp(-eta))
            y = rng.binomial(n=1, p=p, size=n_j)

            center_labels = np.full(n_j, f"C{j}")
            data: dict[str, Any] = {
                "center": center_labels,
                "y": y,
            }
            for k in range(self.n_features):
                data[f"x{k+1}"] = X_cov[:, k]

            rows.append(pd.DataFrame(data))

        df = pd.concat(rows, ignore_index=True)
        return df
