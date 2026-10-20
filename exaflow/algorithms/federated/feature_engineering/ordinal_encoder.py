from __future__ import annotations

from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd

from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.utils.agg_client import AggregationClient
from exaflow.algorithms.federated.utils.interfaces import FederatedTransformer
from exaflow.algorithms.federated.validation import as_scalar
from exaflow.algorithms.federated.validation import check_category_order
from exaflow.algorithms.federated.validation import check_choice
from exaflow.algorithms.federated.validation import check_dataframe


class FederatedOrdinalEncoder(FederatedTransformer):
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
        self.unknown_value = as_scalar(unknown_value, int, name="unknown_value")

    def fit(
        self,
        *,
        agg_client: AggregationClient,
        data: pd.DataFrame,
        categorical_vars: List[str],
        numerical_vars: Optional[List[str]] = None,
    ) -> None:
        """Set categories from provided metadata; aggregation is not supported."""
        check_dataframe(data)
        check_choice(self.handle_unknown, ("ignore", "error"), name="handle_unknown")
        if not self._preset_categories:
            raise BadInputError(
                "FederatedOrdinalEncoder requires categories to be provided "
                "via the constructor."
            )
        for var in categorical_vars:
            check_category_order(self._preset_categories.get(var, []))
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
        check_dataframe(data)
        check_choice(self.handle_unknown, ("ignore", "error"), name="handle_unknown")
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
            dtype = check_category_order(categories)
            cat = pd.Categorical(col, dtype=dtype)
            codes = cat.codes
            if self.handle_unknown == "error" and np.any(codes < 0):
                raise BadInputError(
                    f"Unknown categories encountered in column '{var}'."
                )
            if self.handle_unknown == "ignore":
                codes = np.where(codes < 0, self.unknown_value, codes)
            encoded[:, idx] = codes

        return encoded
