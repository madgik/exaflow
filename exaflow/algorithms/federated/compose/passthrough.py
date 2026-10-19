from __future__ import annotations

from typing import List
from typing import Optional

import numpy as np
import pandas as pd

from exaflow.algorithms.federated.utils.agg_client import AggregationClient
from exaflow.algorithms.federated.utils.interfaces import FederatedTransformer


class FederatedPassthrough(FederatedTransformer):
    """Pass-through transformer for selected columns."""

    def fit(
        self,
        *,
        agg_client: AggregationClient,
        data: pd.DataFrame,
        categorical_vars: List[str],
        numerical_vars: Optional[List[str]] = None,
    ) -> None:
        return None

    def get_feature_names_out(
        self,
        *,
        categorical_vars: List[str],
        numerical_vars: Optional[List[str]] = None,
    ) -> List[str]:
        return list(numerical_vars or [])

    def transform(
        self,
        data: pd.DataFrame,
        *,
        categorical_vars: List[str],
        numerical_vars: Optional[List[str]] = None,
    ) -> np.ndarray:
        numerical_vars = list(numerical_vars or [])
        if not numerical_vars:
            return np.empty((len(data), 0), dtype=float)

        cols = []
        for var in numerical_vars:
            if var not in data.columns:
                cols.append(np.zeros((len(data),), dtype=float))
                continue

            col = data[var]
            if isinstance(col, pd.DataFrame):
                col = col.iloc[:, 0]
            cols.append(col.to_numpy(dtype=float, copy=False).reshape(-1))

        return np.column_stack(cols)
