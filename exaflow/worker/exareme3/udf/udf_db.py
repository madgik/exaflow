from __future__ import annotations

from typing import Iterable
from typing import Set

import pyarrow as pa

from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.data_filters import build_filter_clause


def primary_table_name(data_model: str) -> str:
    sanitized = data_model
    for ch in (":", "-", "."):
        sanitized = sanitized.replace(ch, "_")
    return f'"{sanitized}__primary_data"'


def quote_identifier(identifier: str) -> str:
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


def quote_literal(value: str) -> str:
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def load_algorithm_arrow_table(
    inputdata: Inputdata,
    *,
    dropna: bool = True,
    include_dataset: bool = False,
    extra_columns: Iterable[str] | None = None,
) -> pa.Table:
    """
    Shared data loader for exaflow algorithms, returning an Arrow Table.
    """
    required_columns: Set[str] = set(inputdata.x or []) | set(inputdata.y or [])
    if include_dataset:
        required_columns.add("dataset")
    if extra_columns:
        required_columns.update(extra_columns)

    return _fetch_with_duckdb(inputdata, required_columns, dropna=dropna)


def _fetch_with_duckdb(
    inputdata: Inputdata, required_columns: Set[str], *, dropna: bool
) -> pa.Table:
    import duckdb

    from exaflow.worker import config as worker_config

    datasets = (inputdata.datasets or []) + (
        inputdata.validation_datasets if inputdata.validation_datasets else []
    )

    table_name = primary_table_name(inputdata.data_model)
    columns = sorted(required_columns) if required_columns else ["*"]
    select_columns = ", ".join(quote_identifier(col) for col in columns)

    where_clauses = []
    datasets_clause = ", ".join(quote_literal(value) for value in datasets)
    where_clauses.append(f"{quote_identifier('dataset')} IN ({datasets_clause})")

    if inputdata.filters:
        where_clauses.append(build_filter_clause(inputdata.filters))

    if dropna and required_columns:
        not_null_clause = " AND ".join(
            f"{quote_identifier(col)} IS NOT NULL" for col in required_columns
        )
        where_clauses.append(not_null_clause)

    where_sql = ""
    if where_clauses:
        where_sql = " WHERE " + " AND ".join(f"({clause})" for clause in where_clauses)

    query = f"SELECT {select_columns} FROM {table_name}{where_sql}"

    with duckdb.connect(worker_config.duckdb.path, read_only=True) as conn:
        # Use fetch_arrow_table for zero-copy loading
        arrow_table = conn.execute(query).fetch_arrow_table()

    # Drop duplicate column names if any (Arrow tables enforce unique names usually, but good to be safe)
    # Note: Arrow handles duplicates differently than Pandas, usually by appending suffixes or erroring.
    # DuckDB should return unique columns if the query is well-formed.

    return arrow_table
