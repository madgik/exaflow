from __future__ import annotations

from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd

from exaflow.algorithms.federated.agg_client import AggregationClient
from exaflow.algorithms.federated.interfaces import FederatedPreprocessor


class FederatedOrdinalEncoder(FederatedPreprocessor):
    """
    Ordinal encoder for federated categorical features that enforces a fixed
    category ordering from metadata.

    This mirrors sklearn's OrdinalEncoder configured with explicit categories:
    the encoder does not learn categories from data; it only validates/encodes
    inputs against the provided order. Unknown values are either rejected
    (`handle_unknown="error"`) or mapped to `unknown_value` (default `-1`).
    """

    def __init__(
        self,
        *,
        categories: Optional[Dict[str, List]] = None,
        handle_unknown: str = "ignore",
        unknown_value: int = -1,
    ) -> None:
        self._preset_categories = categories or {}
        self.categories_: Dict[str, List] = {}
        self.handle_unknown = handle_unknown
        self.unknown_value = int(unknown_value)

    def fit(
        self,
        *,
        agg_client: AggregationClient,
        data: pd.DataFrame,
        categorical_vars: List[str],
        numerical_vars: Optional[List[str]] = None,
    ) -> None:
        """Set categories from provided metadata; aggregation is not supported."""
        if not self._preset_categories:
            raise ValueError(
                "FederatedOrdinalEncoder requires categories to be provided "
                "via the constructor."
            )
        self.categories_ = {
            var: list(self._preset_categories.get(var, [])) for var in categorical_vars
        }

    def get_feature_names_out(
        self,
        *,
        categorical_vars: List[str],
        numerical_vars: Optional[List[str]] = None,
    ) -> List[str]:
        numerical_vars = numerical_vars or []
        return list(categorical_vars) + list(numerical_vars)

    def transform(
        self,
        data: pd.DataFrame,
        *,
        categorical_vars: List[str],
        numerical_vars: Optional[List[str]] = None,
    ) -> np.ndarray:
        n_rows = len(data)
        n_cat = len(categorical_vars)
        encoded = np.empty((n_rows, n_cat), dtype=int)

        for idx, var in enumerate(categorical_vars):
            categories = self.categories_.get(var, [])
            if var not in data.columns:
                encoded[:, idx] = self.unknown_value
                continue

            col = data[var]
            if isinstance(col, pd.DataFrame):
                col = col.iloc[:, 0]
            cat = pd.Categorical(col, categories=categories)
            codes = cat.codes
            if self.handle_unknown == "error" and np.any(codes < 0):
                raise ValueError(f"Unknown categories encountered in column '{var}'.")
            if self.handle_unknown == "ignore":
                codes = np.where(codes < 0, self.unknown_value, codes)
            elif self.handle_unknown != "error":
                raise ValueError(
                    "handle_unknown must be 'ignore' or 'error', "
                    f"got '{self.handle_unknown}'."
                )
            encoded[:, idx] = codes

        return encoded
