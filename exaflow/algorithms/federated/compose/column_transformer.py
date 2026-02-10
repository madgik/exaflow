from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from typing import Iterable
from typing import List
from typing import Literal
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
import pandas as pd

from exaflow.algorithms.federated.preprocessing import FederatedPassthrough
from exaflow.algorithms.federated.utils.agg_client import AggregationClient
from exaflow.algorithms.federated.utils.interfaces import FederatedTransformer

ColumnSelector = Callable[[pd.DataFrame], Sequence[str]]
ColumnSpec = Sequence[str] | Sequence[int] | Sequence[bool] | str | ColumnSelector
TransformerSpec = Tuple[str, FederatedTransformer, ColumnSpec]
RemainderSpec = Literal["drop", "passthrough"]


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
        remainder: RemainderSpec = "drop",
    ) -> None:
        self.transformers = list(transformers)
        self.prefix_feature_names = prefix_feature_names
        self.remainder = remainder
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
        used_cat = set()
        used_num = set()
        for name, transformer, columns in self.transformers:
            cat_vars, num_vars = self._resolve_columns(
                columns,
                data=data,
                categorical_vars=categorical_vars,
                numerical_vars=numerical_vars,
            )
            used_cat.update(cat_vars)
            used_num.update(num_vars)
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

        if self.remainder == "passthrough":
            remaining_cat = [v for v in categorical_vars if v not in used_cat]
            if remaining_cat:
                raise ValueError(
                    "FederatedColumnTransformer remainder passthrough does not "
                    f"support categorical vars: {remaining_cat}"
                )
            remaining_num = [v for v in numerical_vars if v not in used_num]
            if remaining_num:
                passthrough = FederatedPassthrough()
                passthrough.fit(
                    agg_client=agg_client,
                    data=data,
                    categorical_vars=[],
                    numerical_vars=remaining_num,
                )
                self._resolved.append(
                    _ResolvedTransformer(
                        name="remainder",
                        transformer=passthrough,
                        categorical_vars=[],
                        numerical_vars=remaining_num,
                    )
                )
        elif self.remainder != "drop":
            raise ValueError(
                "FederatedColumnTransformer remainder must be 'drop' or "
                f"'passthrough', got {self.remainder!r}"
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
        data: pd.DataFrame,
        categorical_vars: List[str],
        numerical_vars: List[str],
    ) -> Tuple[List[str], List[str]]:
        if isinstance(columns, str):
            return [], [columns]

        if callable(columns):
            columns = columns(data)

        columns = list(columns)
        if columns and all(isinstance(col, bool) for col in columns):
            if len(columns) != len(data.columns):
                raise ValueError(
                    "Boolean column mask length must match number of columns."
                )
            columns = list(data.columns[np.asarray(columns, dtype=bool)])
        elif columns and all(isinstance(col, int) for col in columns):
            columns = list(data.columns[np.asarray(columns, dtype=int)])

        cat_set = set(categorical_vars)
        num_set = set(numerical_vars)
        cat_cols = [c for c in columns if c in cat_set]
        num_cols = [c for c in columns if c in num_set or c not in cat_set]
        return cat_cols, num_cols


def make_column_selector(
    *,
    dtype_include: Optional[Sequence[str | np.dtype]] = None,
    dtype_exclude: Optional[Sequence[str | np.dtype]] = None,
) -> ColumnSelector:
    """Return a callable column selector similar to sklearn."""

    def _matches(dtype: np.dtype, spec: str | np.dtype) -> bool:
        if spec in ("category", "categorical"):
            return pd.api.types.is_categorical_dtype(dtype)
        if spec in ("object", object):
            return pd.api.types.is_object_dtype(dtype)
        return np.issubdtype(dtype, spec)

    def _selector(df: pd.DataFrame) -> List[str]:
        include = list(dtype_include or [])
        exclude = list(dtype_exclude or [])
        dtypes = df.dtypes
        if include:
            mask = dtypes.apply(
                lambda dtype: any(_matches(dtype, inc) for inc in include)
            )
        else:
            mask = pd.Series(True, index=dtypes.index)
        if exclude:
            exclude_mask = dtypes.apply(
                lambda dtype: any(_matches(dtype, exc) for exc in exclude)
            )
            mask = mask & ~exclude_mask
        return dtypes[mask].index.tolist()

    return _selector
