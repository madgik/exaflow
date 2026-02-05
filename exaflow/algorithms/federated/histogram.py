from __future__ import annotations

from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np
import pandas as pd
from pandas.api import types as pd_types

HistogramBin = Union[float, str]


class GroupedHistogram:
    def __init__(self, groups: List[HistogramBin], counts: List[List[Optional[int]]]):
        self.groups = groups
        self.counts = counts


class HistogramResult:
    def __init__(
        self,
        bins: List[HistogramBin],
        counts: List[Optional[int]],
        grouped: Dict[str, GroupedHistogram],
    ):
        self.bins = bins
        self.counts = counts
        self.grouped = grouped

    def as_payload(self) -> Dict[str, object]:
        return {
            "bins": self.bins,
            "counts": self.counts,
            "grouped": {
                name: {"groups": grouped.groups, "counts": grouped.counts}
                for name, grouped in self.grouped.items()
            },
        }


class FederatedHistogram:
    """
    Statsmodels-style histogram computation for federated data.

    ``hist`` returns a ``HistogramResult`` with bin edges or category levels,
    counts masked by ``min_row_count``, and grouped counts per ``x_vars`` when
    provided. The API is intentionally thin and relies on aggregation primitives.
    """

    def __init__(self, agg_client):
        self.agg_client = agg_client

    def hist(
        self,
        *,
        data: pd.DataFrame,
        y_var: str,
        x_vars: List[str],
        metadata: Dict,
        bins: int,
        min_row_count: int,
    ) -> HistogramResult:
        y_meta = metadata.get(y_var, {})
        is_categorical = y_meta.get("is_categorical", False)

        if is_categorical:
            return self._categorical_histogram(
                data=data,
                y_var=y_var,
                x_vars=x_vars,
                metadata=metadata,
                min_row_count=min_row_count,
            )

        return self._numerical_histogram(
            data=data,
            y_var=y_var,
            x_vars=x_vars,
            metadata=metadata,
            bins=bins,
            min_row_count=min_row_count,
        )

    def _value_counts(self, series: pd.Series, categories) -> List[int]:
        counts = series.value_counts()
        resolved = []
        for cat in categories:
            count = counts.get(cat, 0)
            if count == 0 and isinstance(cat, str):
                count = self._value_counts_numeric_fallback(counts, cat)
            resolved.append(int(count))
        return resolved

    def _value_counts_numeric_fallback(self, counts, cat: str) -> int:
        for caster in (float, int):
            try:
                coerced = caster(cat)
            except (TypeError, ValueError):
                continue
            value = counts.get(coerced, 0)
            if value:
                return value
        return 0

    def _normalize_categories(
        self, categories: List[HistogramBin], series: pd.Series
    ) -> List[HistogramBin]:
        dtype = series.dtype
        if pd_types.is_integer_dtype(dtype):

            def caster(value):
                return int(float(value))

        elif pd_types.is_float_dtype(dtype):

            def caster(value):
                return float(value)

        else:
            return categories

        normalized: List[HistogramBin] = []
        for value in categories:
            if value is None:
                normalized.append(value)
                continue
            try:
                normalized.append(caster(value))
            except (TypeError, ValueError):
                normalized.append(value)
        return normalized

    def _get_enumerations(self, meta_entry, series: pd.Series) -> List[HistogramBin]:
        if meta_entry and meta_entry.get("enumerations"):
            categories = list(meta_entry["enumerations"].keys())
            return self._normalize_categories(categories, series)
        return sorted(series.dropna().unique().tolist())

    def _mask_counts(
        self, values: Sequence[float], min_row_count: int
    ) -> List[Optional[int]]:
        masked = []
        for value in values:
            count = int(round(value))
            masked.append(count if count >= min_row_count else None)
        return masked

    def _aggregate_matrix(self, matrix: Sequence[Sequence[int]]) -> List[List[float]]:
        if not matrix:
            return []
        arr = np.asarray(matrix, dtype=float)
        if arr.size == 0:
            return arr.tolist()
        flat = arr.reshape(-1).tolist()
        summed = self.agg_client.sum(flat)
        return np.asarray(summed).reshape(arr.shape).tolist()

    def _categorical_histogram(
        self,
        *,
        data: pd.DataFrame,
        y_var: str,
        x_vars: List[str],
        metadata: Dict,
        min_row_count: int,
    ) -> HistogramResult:
        series = data[y_var]
        categories = self._get_enumerations(metadata.get(y_var), series)
        local_counts = self._value_counts(series, categories)
        global_counts = self.agg_client.sum(local_counts)
        masked_counts = self._mask_counts(global_counts, min_row_count)

        grouped: Dict[str, GroupedHistogram] = {}
        for x_var in x_vars:
            x_series = data[x_var]
            groups = self._get_enumerations(metadata.get(x_var), x_series)
            matrix = [
                self._value_counts(
                    data.loc[x_series == group, y_var],
                    categories,
                )
                for group in groups
            ]
            global_matrix = self._aggregate_matrix(matrix)
            grouped[x_var] = GroupedHistogram(
                groups=groups,
                counts=[self._mask_counts(row, min_row_count) for row in global_matrix],
            )

        return HistogramResult(
            bins=categories,
            counts=masked_counts,
            grouped=grouped,
        )

    def _numerical_histogram(
        self,
        *,
        data: pd.DataFrame,
        y_var: str,
        x_vars: List[str],
        metadata: Dict,
        bins: int,
        min_row_count: int,
    ) -> HistogramResult:
        bins = max(1, int(round(bins)))
        values = data[y_var].to_numpy(dtype=float, copy=False)
        n_obs = int(values.size)
        total_n_obs_arr = self.agg_client.sum(np.array([float(n_obs)], dtype=float))
        global_min_arr = self.agg_client.min(
            np.array([float(np.min(values)) if n_obs else np.inf], dtype=float)
        )
        global_max_arr = self.agg_client.max(
            np.array([float(np.max(values)) if n_obs else -np.inf], dtype=float)
        )
        total_n_obs = total_n_obs_arr[0]
        if total_n_obs == 0:
            raise ValueError("No data available to compute histogram.")

        global_min = float(global_min_arr[0])
        global_max = float(global_max_arr[0])
        if not np.isfinite(global_min) or not np.isfinite(global_max):
            raise ValueError("Unable to determine histogram bounds.")
        if global_min == global_max:
            global_max = global_min + 1.0

        bin_edges = np.linspace(global_min, global_max, bins + 1)
        local_hist, _ = np.histogram(values, bins=bin_edges)
        global_hist = self.agg_client.sum(local_hist)
        masked_counts = self._mask_counts(global_hist, min_row_count)

        grouped: Dict[str, GroupedHistogram] = {}
        for x_var in x_vars:
            groups = self._get_enumerations(metadata.get(x_var), data[x_var])
            matrix = []
            for group in groups:
                subset = data.loc[data[x_var] == group, y_var].to_numpy(
                    dtype=float, copy=False
                )
                group_hist, _ = np.histogram(subset, bins=bin_edges)
                matrix.append(group_hist.tolist())
            global_matrix = self._aggregate_matrix(matrix)
            grouped[x_var] = GroupedHistogram(
                groups=groups,
                counts=[self._mask_counts(row, min_row_count) for row in global_matrix],
            )

        return HistogramResult(
            bins=bin_edges.tolist(),
            counts=masked_counts,
            grouped=grouped,
        )
