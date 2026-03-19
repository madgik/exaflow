from __future__ import annotations

from collections.abc import Callable
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd

from exaflow.algorithms.federated.utils.aggregators.numpy_aggregator import (
    NumpyAggregator,
)


class FederatedGroupBy:
    """Fluent interface for applying group-by aggregations on a local DataFrame.

    Designed to be used on each worker's data partition; federated aggregation
    functions (e.g. from :class:`NumpyAggregator`) should be passed as the
    ``aggregate_function`` so that partial results are coordinated across all
    workers.

    Example::

        agg = NumpyAggregator(agg_client)
        gb = FederatedGroupBy(df, agg)
        result = gb.group_by("category").aggregate("mean_x", agg.global_avg, "x")()
    """

    class Expression:
        def __init__(
            self,
            df: pd.DataFrame,
            aggregator: NumpyAggregator,
            where_expression: Optional[str] = None,
        ) -> None:
            self.array: pd.DataFrame = df
            self.aggregator: NumpyAggregator = aggregator
            self.where: Optional[str] = where_expression
            self.grouping_columns: Optional[List[str]] = None
            self.aggregations: List[Tuple[str, Callable, List[str]]] = []

        def group_by(self, *grouping_columns: str) -> "FederatedGroupBy.Expression":
            self.grouping_columns = list(grouping_columns) if grouping_columns else None
            return self

        def aggregate(
            self,
            alias: str,
            aggregate_function: Callable,
            *columns: str,
        ) -> "FederatedGroupBy.Expression":
            self.aggregations.append((alias, aggregate_function, list(columns)))
            return self

        def __call__(self) -> pd.DataFrame:
            # Apply where filter if specified
            if self.where:
                df_filtered = self.array.query(self.where)
            else:
                df_filtered = self.array.copy()

            # If no aggregations, return filtered dataframe
            if not self.aggregations:
                return df_filtered

            # Case 1: Global aggregation (no group-by)
            if not self.grouping_columns:
                result = {}
                for alias, agg_func, columns in self.aggregations:
                    col_data = [
                        df_filtered[col].to_numpy(dtype=float) for col in columns
                    ]
                    # If local data is empty, use NaN sentinel to keep sync
                    if len(df_filtered) == 0:
                        col_data = [np.array([np.nan], dtype=float) for _ in columns]
                    result[alias] = agg_func(*col_data)
                return pd.DataFrame([result])

            # Case 2: Federated Group-By
            # -----------------------------------------------------------------
            # FEDERATED SYNC STEP:
            # Get the union of all group keys across all clients to ensure
            # all workers iterate in the same order and stay in lock-step.
            # -----------------------------------------------------------------
            if len(self.grouping_columns) > 1:
                # Compound key (tuple)
                group_key_series = df_filtered[self.grouping_columns].apply(
                    tuple, axis=1
                )
            else:
                # Single column key
                group_key_series = df_filtered[self.grouping_columns[0]]

            # Identify local keys (ignoring NaNs)
            local_keys = group_key_series.unique()
            # Drop entries that are truly null
            local_keys = local_keys[~pd.isna(local_keys)]

            # Synchronize with other clients to get the global union of keys
            global_keys = sorted(
                self.aggregator.fed_union(local_keys), key=lambda k: str(k)
            )

            final_results = []
            for key in global_keys:
                # Filter local data for this specific global key
                if len(self.grouping_columns) > 1:
                    mask = (
                        df_filtered[self.grouping_columns].apply(tuple, axis=1) == key
                    )
                    group_data = df_filtered[mask]
                    res_row = {
                        self.grouping_columns[i]: key[i]
                        for i in range(len(self.grouping_columns))
                    }
                else:
                    group_data = df_filtered[
                        df_filtered[self.grouping_columns[0]] == key
                    ]
                    res_row = {self.grouping_columns[0]: key}

                # Apply each federated aggregation function for this group
                for alias, agg_func, columns in self.aggregations:
                    inputs = [group_data[col].to_numpy(dtype=float) for col in columns]

                    # If this client has no data for this group, we must still
                    # participate with a NaN sentinel to avoid deadlock and
                    # allow aggregators to correctly ignore this client.
                    if len(group_data) == 0:
                        inputs = [np.array([np.nan], dtype=float) for _ in columns]

                    res_row[alias] = agg_func(*inputs)
                final_results.append(res_row)

            # Handle edge case: no groups found globally
            if not final_results:
                cols = self.grouping_columns + [agg[0] for agg in self.aggregations]
                return pd.DataFrame(columns=cols).set_index(self.grouping_columns)

            # Assemble into a DataFrame indexed by the grouping columns
            return pd.DataFrame(final_results).set_index(self.grouping_columns)

    def __init__(self, df: pd.DataFrame, aggregator: NumpyAggregator) -> None:
        self.array: pd.DataFrame = df
        self.aggregator: NumpyAggregator = aggregator

    def _new_expression(
        self, where: Optional[str] = None
    ) -> "FederatedGroupBy.Expression":
        return FederatedGroupBy.Expression(self.array, self.aggregator, where)

    def where(self, where_expression: str) -> "FederatedGroupBy.Expression":
        return self._new_expression(where_expression)

    def group_by(self, *grouping_columns: str) -> "FederatedGroupBy.Expression":
        return self._new_expression().group_by(*grouping_columns)

    def aggregate(
        self,
        alias: str,
        aggregate_function: Callable,
        *columns: str,
    ) -> "FederatedGroupBy.Expression":
        return self._new_expression().aggregate(alias, aggregate_function, *columns)
