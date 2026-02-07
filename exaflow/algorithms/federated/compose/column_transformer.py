from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
import pandas as pd

from exaflow.algorithms.federated.utils.agg_client import AggregationClient
from exaflow.algorithms.federated.utils.interfaces import FederatedTransformer

ColumnSpec = Sequence[str] | str
TransformerSpec = Tuple[str, FederatedTransformer, ColumnSpec]


@dataclass
class _ResolvedTransformer:
    name: str
    transformer: FederatedTransformer
    categorical_vars: List[str]
    numerical_vars: List[str]


class FederatedColumnTransformer(FederatedTransformer):
    """Apply transformers to column subsets like sklearn ColumnTransformer."""

    def __init__(
        self,
        transformers: Iterable[TransformerSpec],
        *,
        prefix_feature_names: bool = False,
    ) -> None:
        self.transformers = list(transformers)
        self.prefix_feature_names = prefix_feature_names
        self._resolved: List[_ResolvedTransformer] = []

    def fit(
        self,
        *,
        agg_client: AggregationClient,
        data: pd.DataFrame,
        categorical_vars: List[str],
        numerical_vars: Optional[List[str]] = None,
    ) -> None:
        numerical_vars = list(numerical_vars or [])
        self._resolved = []
        for name, transformer, columns in self.transformers:
            cat_vars, num_vars = self._resolve_columns(
                columns,
                categorical_vars=categorical_vars,
                numerical_vars=numerical_vars,
            )
            transformer.fit(
                agg_client=agg_client,
                data=data,
                categorical_vars=cat_vars,
                numerical_vars=num_vars,
            )
            self._resolved.append(
                _ResolvedTransformer(
                    name=name,
                    transformer=transformer,
                    categorical_vars=cat_vars,
                    numerical_vars=num_vars,
                )
            )

    def get_feature_names_out(
        self,
        *,
        categorical_vars: List[str],
        numerical_vars: Optional[List[str]] = None,
    ) -> List[str]:
        features: List[str] = []
        for resolved in self._resolved:
            names = resolved.transformer.get_feature_names_out(
                categorical_vars=resolved.categorical_vars,
                numerical_vars=resolved.numerical_vars,
            )
            if self.prefix_feature_names:
                features.extend([f"{resolved.name}__{name}" for name in names])
            else:
                features.extend(names)
        return features

    def transform(
        self,
        data: pd.DataFrame,
        *,
        categorical_vars: List[str],
        numerical_vars: Optional[List[str]] = None,
    ) -> np.ndarray:
        blocks = []
        for resolved in self._resolved:
            block = resolved.transformer.transform(
                data,
                categorical_vars=resolved.categorical_vars,
                numerical_vars=resolved.numerical_vars,
            )
            blocks.append(np.asarray(block, dtype=float))

        if not blocks:
            return np.empty((len(data), 0), dtype=float)
        return np.column_stack(blocks)

    @staticmethod
    def _resolve_columns(
        columns: ColumnSpec,
        *,
        categorical_vars: List[str],
        numerical_vars: List[str],
    ) -> Tuple[List[str], List[str]]:
        if isinstance(columns, str):
            if columns == "categorical":
                return list(categorical_vars), []
            if columns == "numerical":
                return [], list(numerical_vars)
            return [], [columns]

        columns = list(columns)
        cat_set = set(categorical_vars)
        num_set = set(numerical_vars)
        cat_cols = [c for c in columns if c in cat_set]
        num_cols = [c for c in columns if c in num_set or c not in cat_set]
        return cat_cols, num_cols
