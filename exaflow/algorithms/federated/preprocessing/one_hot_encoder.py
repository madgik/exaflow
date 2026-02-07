from __future__ import annotations

from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd

from exaflow.algorithms.federated.utils.agg_client import AggregationClient
from exaflow.algorithms.federated.utils.interfaces import FederatedTransformer


class FederatedOneHotEncoder(FederatedTransformer):
    """Federated one-hot encoder without intercept support."""

    def __init__(self) -> None:
        self.dummy_categories: Dict[str, List] = {}

    def fit(
        self,
        *,
        agg_client: AggregationClient,
        data: pd.DataFrame,
        categorical_vars: List[str],
        numerical_vars: Optional[List[str]] = None,
    ) -> None:
        if not categorical_vars:
            self.dummy_categories = {}
            return

        local_levels = self._collect_categorical_levels_from_df(data, categorical_vars)
        merged: Dict[str, List] = {}
        for var in categorical_vars:
            levels = local_levels.get(var, [])
            global_levels = agg_client.union([lvl for lvl in levels if lvl is not None])
            merged[var] = sorted(global_levels)

        self.dummy_categories = {var: levels[1:] for var, levels in merged.items()}

    def get_feature_names_out(
        self,
        *,
        categorical_vars: List[str],
        numerical_vars: Optional[List[str]] = None,
    ) -> List[str]:
        numerical_vars = numerical_vars or []
        labels: List[str] = []
        for var in categorical_vars:
            labels.extend(
                [f"{var}[{lvl}]" for lvl in self.dummy_categories.get(var, [])]
            )
        labels.extend(numerical_vars)
        return labels

    def transform(
        self,
        data: pd.DataFrame,
        *,
        categorical_vars: List[str],
        numerical_vars: Optional[List[str]] = None,
    ) -> np.ndarray:
        numerical_vars = numerical_vars or []
        return self._build_design_matrix(
            data,
            categorical_vars=categorical_vars,
            dummy_categories=self.dummy_categories,
            numerical_vars=numerical_vars,
        )

    @staticmethod
    def _collect_categorical_levels_from_df(
        data: pd.DataFrame, categorical_vars: List[str]
    ) -> Dict[str, List]:
        """Collect observed levels per categorical variable from a local DataFrame."""
        levels: Dict[str, List] = {}
        for var in categorical_vars:
            if var not in data.columns:
                levels[var] = []
                continue

            col = data[var]
            if isinstance(col, pd.DataFrame):
                col = col.iloc[:, 0]

            vals = col.dropna().unique().tolist()
            levels[var] = vals
        return levels

    @staticmethod
    def _build_design_matrix(
        data: pd.DataFrame,
        *,
        categorical_vars: List[str],
        dummy_categories: Dict[str, List],
        numerical_vars: List[str],
    ) -> np.ndarray:
        n_rows = len(data)
        n_dummy_cols = sum(
            len(dummy_categories.get(var, [])) for var in categorical_vars
        )
        total_cols = n_dummy_cols + len(numerical_vars)
        design = np.empty((n_rows, total_cols), dtype=float)

        col_idx = 0

        for var in categorical_vars:
            categories = dummy_categories.get(var, [])
            if var not in data.columns:
                if categories:
                    design[:, col_idx : col_idx + len(categories)] = 0.0
                    col_idx += len(categories)
                continue

            col = data[var]
            if isinstance(col, pd.DataFrame):
                col = col.iloc[:, 0]
            values = col
            for category in categories:
                encoded = (
                    (values == category).to_numpy(dtype=float, copy=False).reshape(-1)
                )
                design[:, col_idx] = encoded
                col_idx += 1

        for var in numerical_vars:
            if var not in data.columns:
                design[:, col_idx] = 0.0
                col_idx += 1
                continue

            col = data[var]
            if isinstance(col, pd.DataFrame):
                col = col.iloc[:, 0]

            num_col = col.to_numpy(dtype=float, copy=False).reshape(-1)
            design[:, col_idx] = num_col
            col_idx += 1

        return design
