from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Callable
from typing import Optional
from typing import Protocol

import numpy as np
import pandas as pd

from exaflow.algorithms.federated.utils.aggregators.numpy_aggregator import (
    NumpyAggregator,
)

AggregationFunction = Callable[..., object]
AggregationSpec = tuple[str, AggregationFunction, list[str]]


@dataclass(frozen=True)
class FederatedSQLResults:
    dataframe: pd.DataFrame


class FederatedSQLAfterWhere(Protocol):
    def group_by(self, *grouping_columns: str) -> FederatedSQLAfterGroupBy: ...

    def aggregate(
        self, alias: str, aggregate_function: AggregationFunction, *columns: str
    ) -> FederatedSQLAfterAggregate: ...


class FederatedSQLAfterGroupBy(Protocol):
    def aggregate(
        self, alias: str, aggregate_function: AggregationFunction, *columns: str
    ) -> FederatedSQLAfterAggregate: ...


class FederatedSQLAfterAggregate(Protocol):
    def aggregate(
        self, alias: str, aggregate_function: AggregationFunction, *columns: str
    ) -> FederatedSQLAfterAggregate: ...

    def run(self) -> FederatedSQLResults: ...


class FederatedSQLStart(Protocol):
    def where(self, where_expression: str) -> FederatedSQLAfterWhere: ...

    def group_by(self, *grouping_columns: str) -> FederatedSQLAfterGroupBy: ...

    def aggregate(
        self, alias: str, aggregate_function: AggregationFunction, *columns: str
    ) -> FederatedSQLAfterAggregate: ...


@dataclass
class _QueryState:
    dataframe: pd.DataFrame
    aggregator: NumpyAggregator
    where_expression: Optional[str] = None
    grouping_columns: Optional[list[str]] = None
    aggregations: list[AggregationSpec] = field(default_factory=list)
    has_run: bool = False

    def clone(self) -> _QueryState:
        return _QueryState(
            dataframe=self.dataframe,
            aggregator=self.aggregator,
            where_expression=self.where_expression,
            grouping_columns=None
            if self.grouping_columns is None
            else list(self.grouping_columns),
            aggregations=[
                (alias, function, list(columns))
                for alias, function, columns in self.aggregations
            ],
            has_run=self.has_run,
        )


class _BaseStep:
    def __init__(self, state: _QueryState):
        self._state = state

    def _with_where(self, where_expression: str) -> _QueryState:
        next_state = self._state.clone()
        next_state.where_expression = where_expression
        return next_state

    def _with_group_by(self, grouping_columns: tuple[str, ...]) -> _QueryState:
        next_state = self._state.clone()
        next_state.grouping_columns = (
            list(grouping_columns) if grouping_columns else None
        )
        return next_state

    def _with_aggregate(
        self,
        alias: str,
        aggregate_function: AggregationFunction,
        columns: tuple[str, ...],
    ) -> _QueryState:
        next_state = self._state.clone()
        next_state.aggregations.append((alias, aggregate_function, list(columns)))
        return next_state


class FederatedSQL(FederatedSQLStart, _BaseStep):
    def __init__(self, dataframe: pd.DataFrame, aggregator: NumpyAggregator):
        _BaseStep.__init__(
            self,
            _QueryState(dataframe=dataframe, aggregator=aggregator),
        )

    def where(self, where_expression: str) -> FederatedSQLAfterWhere:
        return _AfterWhereStep(self._with_where(where_expression))

    def group_by(self, *grouping_columns: str) -> FederatedSQLAfterGroupBy:
        return _AfterGroupByStep(self._with_group_by(grouping_columns))

    def aggregate(
        self,
        alias: str,
        aggregate_function: AggregationFunction,
        *columns: str,
    ) -> FederatedSQLAfterAggregate:
        return _AfterAggregateStep(
            self._with_aggregate(alias, aggregate_function, columns)
        )


class _AfterWhereStep(_BaseStep, FederatedSQLAfterWhere):
    def group_by(self, *grouping_columns: str) -> FederatedSQLAfterGroupBy:
        return _AfterGroupByStep(self._with_group_by(grouping_columns))

    def aggregate(
        self,
        alias: str,
        aggregate_function: AggregationFunction,
        *columns: str,
    ) -> FederatedSQLAfterAggregate:
        return _AfterAggregateStep(
            self._with_aggregate(alias, aggregate_function, columns)
        )


class _AfterGroupByStep(_BaseStep, FederatedSQLAfterGroupBy):
    def aggregate(
        self,
        alias: str,
        aggregate_function: AggregationFunction,
        *columns: str,
    ) -> FederatedSQLAfterAggregate:
        return _AfterAggregateStep(
            self._with_aggregate(alias, aggregate_function, columns)
        )


class _AfterAggregateStep(_BaseStep, FederatedSQLAfterAggregate):
    def aggregate(
        self,
        alias: str,
        aggregate_function: AggregationFunction,
        *columns: str,
    ) -> FederatedSQLAfterAggregate:
        return _AfterAggregateStep(
            self._with_aggregate(alias, aggregate_function, columns)
        )

    def run(self) -> FederatedSQLResults:
        if self._state.has_run:
            raise RuntimeError(
                "run() can only be executed once per query builder object."
            )
        if not self._state.aggregations:
            raise RuntimeError("At least one aggregate() must be defined before run().")
        self._state.has_run = True
        return FederatedSQLResults(_execute(self._state))


def _execute(state: _QueryState) -> pd.DataFrame:
    if state.where_expression:
        df_filtered = state.dataframe.query(state.where_expression)
    else:
        df_filtered = state.dataframe.copy()

    if not state.grouping_columns:
        result = {}
        for alias, agg_func, columns in state.aggregations:
            col_data = [df_filtered[col].to_numpy(dtype=float) for col in columns]
            if len(df_filtered) == 0:
                col_data = [np.array([np.nan], dtype=float) for _ in columns]
            result[alias] = agg_func(*col_data)
        return pd.DataFrame([result])

    grouping_columns = state.grouping_columns
    if len(grouping_columns) > 1:
        group_key_series = df_filtered[grouping_columns].apply(tuple, axis=1)
    else:
        group_key_series = df_filtered[grouping_columns[0]]

    local_keys = group_key_series.unique()
    local_keys = local_keys[~pd.isna(local_keys)]
    global_keys = sorted(
        state.aggregator.fed_union(local_keys), key=lambda key: str(key)
    )

    final_results = []
    for key in global_keys:
        if len(grouping_columns) > 1:
            mask = df_filtered[grouping_columns].apply(tuple, axis=1) == key
            group_data = df_filtered[mask]
            res_row = {
                grouping_columns[index]: key[index]
                for index in range(len(grouping_columns))
            }
        else:
            group_data = df_filtered[df_filtered[grouping_columns[0]] == key]
            res_row = {grouping_columns[0]: key}

        for alias, agg_func, columns in state.aggregations:
            inputs = [group_data[col].to_numpy(dtype=float) for col in columns]
            if len(group_data) == 0:
                inputs = [np.array([np.nan], dtype=float) for _ in columns]
            res_row[alias] = agg_func(*inputs)
        final_results.append(res_row)

    if not final_results:
        cols = grouping_columns + [agg[0] for agg in state.aggregations]
        return pd.DataFrame(columns=cols).set_index(grouping_columns)

    return pd.DataFrame(final_results).set_index(grouping_columns)
