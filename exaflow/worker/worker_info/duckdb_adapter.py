from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

import duckdb

from exaflow.worker import config as worker_config

Path(worker_config.duckdb.path).parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=None)
def _read_only_connection(path: str) -> duckdb.DuckDBPyConnection:
    # Opened lazily on the first query, i.e. after the startup loader has closed
    # its read-write connection. DuckDB refuses a read-write connection to a
    # file that this process already holds open read-only, so the loader must
    # not run again in the same process once this connection exists.
    return duckdb.connect(path, read_only=True)


def read_only_cursor() -> duckdb.DuckDBPyConnection:
    """Return a per-call cursor on the process-wide read-only connection."""
    return _read_only_connection(str(worker_config.duckdb.path)).cursor()


def execute_and_fetchall(query: str) -> List:
    with read_only_cursor() as cursor:
        return cursor.execute(query).fetchall()
