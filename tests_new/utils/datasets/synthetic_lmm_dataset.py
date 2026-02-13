from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from tests_new.utils.interfaces.partitioned_table import PartitionedPandasTable


class SyntheticLmmDataset(PartitionedPandasTable):
    """
    Synthetic dataset για LMM (random intercept), για χρήση με FederationTestTemplate.

    Δημιουργεί ένα DataFrame με columns:
      - "center" : κατηγορική μεταβλητή (cluster / site)
      - "y"      : outcome
      - "x1", "x2", ..., "x{n_features}" : covariates

    Η γεννήτρια είναι ουσιαστικά η ίδια λογική με τη synth_lmm_df(...)
    από το test_lmm_reml_fed_vs_cent.py.
    """

    def __init__(
        self,
        n_centers: int = 8,
        n_features: int = 2,
        n_min: int = 30,
        n_max: int = 60,
        beta: Optional[Sequence[float]] = None,
        sigma2: float = 1.2,
        sigma_u2: float = 0.7,
        seed: int = 42,
    ) -> None:
        # 1) Αποθηκεύουμε όλες τις παραμέτρους ΠΡΙΝ το super().__init__()
        self.n_centers = int(n_centers)
        self.n_features = int(n_features)
        self.n_min = int(n_min)
        self.n_max = int(n_max)
        self.beta = None if beta is None else np.asarray(beta, dtype=float)
        self.sigma2 = float(sigma2)
        self.sigma_u2 = float(sigma_u2)
        self.seed = int(seed)

        # 2) Αυτό θα καλέσει get_dataset() μία φορά και θα αποθηκεύσει το global df
        super().__init__()

    def get_dataset(self) -> pd.DataFrame:
        """
        Υλοποίηση του abstract get_dataset() από το PartitionedPandasTable.

        Εδώ παράγουμε ένα single global DataFrame, το οποίο μετά η βάση κλάση
        θα partition-άρει σε local datasets ανά client.
        """
        rng = np.random.default_rng(self.seed)

        if self.beta is None:
            beta = np.array(
                [0.3] + [0.5] * self.n_features,
                dtype=float,
            )
        else:
            beta = self.beta
            if beta.shape[0] != self.n_features + 1:
                raise ValueError(
                    f"beta must have length n_features+1={self.n_features+1} "
                    f"(intercept + slopes), got {beta.shape[0]}"
                )

        rows = []

        for j in range(self.n_centers):
            n_j = int(rng.integers(self.n_min, self.n_max + 1))

            # Χτίζουμε X = [1, x1, x2, ...]
            X_cov = rng.normal(size=(n_j, self.n_features))
            intercept = np.ones((n_j, 1), dtype=float)
            X = np.hstack([intercept, X_cov])

            # Τυχαίο random intercept u_j
            u_j = rng.normal(loc=0.0, scale=np.sqrt(self.sigma_u2))
            # Τυχαίο σφάλμα
            eps = rng.normal(loc=0.0, scale=np.sqrt(self.sigma2), size=n_j)

            # Outcome
            y = X @ beta + u_j + eps

            centers = np.full(n_j, f"C{j}", dtype=object)

            data = {
                "center": centers,
                "y": y,
            }
            for k in range(self.n_features):
                data[f"x{k+1}"] = X_cov[:, k]

            rows.append(pd.DataFrame(data))

        df = pd.concat(rows, ignore_index=True)
        return df
