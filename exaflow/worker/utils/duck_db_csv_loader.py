from __future__ import annotations

import json
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import duckdb

from exaflow.column_names import DATASET_COL
from exaflow.worker import config as worker_config
from exaflow.worker.utils.logger import init_logger

LOGGER = init_logger("WORKER DATA LOADER")


def _sanitize_name(name: str) -> str:
    sanitized = name
    for ch in (":", "-", "."):
        sanitized = sanitized.replace(ch, "_")
    return sanitized


def _reset_metadata_tables(conn: duckdb.DuckDBPyConnection):
    tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
    for table in tables:
        conn.execute(f'DROP TABLE IF EXISTS "{table}"')

    conn.execute(
        """
        CREATE TABLE data_models (
            data_model_id INTEGER,
            code TEXT,
            version TEXT,
            label TEXT,
            status TEXT,
            properties TEXT
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE datasets (
            dataset_id INTEGER,
            data_model_id INTEGER,
            code TEXT,
            label TEXT,
            status TEXT,
            csv_path TEXT,
            properties TEXT
        )
        """
    )


def _collect_all_variables(metadata: dict) -> list[dict]:
    def _walk(node: dict) -> list[dict]:
        collected = list(node.get("variables", []) or [])
        for group in node.get("groups", []) or []:
            collected.extend(_walk(group))
        return collected

    return _walk(metadata)


_SQL_TYPE_TO_DUCKDB = {
    "text": "VARCHAR",
    "int": "INTEGER",
    "real": "DOUBLE",
}


def _build_column_types(metadata: dict) -> dict[str, str]:
    """Map each variable code to its DuckDB column type.

    Categorical variables are always stored as VARCHAR regardless of their
    declared ``sql_type`` so that nominal codes are never silently cast to
    integers or floats by DuckDB's type inference.
    """
    column_types: dict[str, str] = {}
    for var in _collect_all_variables(metadata):
        code = var.get("code")
        if not code:
            continue
        if var.get("isCategorical"):
            column_types[code] = "VARCHAR"
        else:
            sql_type = var.get("sql_type", "text")
            column_types[code] = _SQL_TYPE_TO_DUCKDB.get(sql_type, "VARCHAR")
    return column_types


def _load_variables_metadata(
    conn: duckdb.DuckDBPyConnection, table_prefix: str, metadata: dict
):
    table_name = f"{table_prefix}_variables_metadata"
    conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    conn.execute(f'CREATE TABLE "{table_name}" (code TEXT, metadata TEXT)')
    rows = [
        [variable["code"], json.dumps(_reformat_metadata(variable))]
        for variable in _collect_all_variables(metadata)
    ]
    if rows:
        conn.executemany(f'INSERT INTO "{table_name}" VALUES (?, ?)', rows)


def _build_dataset_properties(csv_columns: list[str], cde_codes: set[str]) -> dict:
    """CSV header columns that are CDEs, in header order."""
    variables = [column for column in csv_columns if column in cde_codes]
    return {"tags": [], "properties": {"variables": variables}}


def _reformat_metadata(metadata: dict) -> dict:
    new_key_assign = {
        "isCategorical": "is_categorical",
        "minValue": "min",
        "maxValue": "max",
    }
    reformatted = dict(metadata)
    for old_key, new_key in new_key_assign.items():
        if old_key in reformatted:
            reformatted[new_key] = reformatted.pop(old_key)

    if "enumerations" in reformatted and isinstance(reformatted["enumerations"], list):
        reformatted["enumerations"] = OrderedDict(
            (
                enumeration.get("code", ""),
                enumeration.get("label", ""),
            )
            for enumeration in reformatted["enumerations"]
        )

    return reformatted


_SOURCE_CSV_COL = "__source_csv_idx"


class PrimaryDataTable(NamedTuple):
    dataset_csv_map: dict[str, Path]
    """Dataset code -> the CSV file that defines it."""
    csv_columns: dict[Path, list[str]]
    """Each CSV's header columns, in header order ([] if unreadable)."""


def _create_primary_data_table(
    conn: duckdb.DuckDBPyConnection,
    table_prefix: str,
    csv_paths: list[Path],
    metadata: dict | None = None,
) -> PrimaryDataTable:
    """Create the primary data table and return what the loader needs from it.

    Each CSV is read once into a staging table tagged with its source index,
    so dataset codes are derived without re-reading the files. The header
    columns read here are returned too, so dataset variables reuse them.
    """
    if not csv_paths:
        raise ValueError("No CSV files provided to create the primary data table.")

    column_types = _build_column_types(metadata) if metadata else None

    table_name = f"{table_prefix}__primary_data"
    conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')

    # Header sniffing has a fixed per-file cost and DuckDB releases the GIL.
    with ThreadPoolExecutor() as pool:
        column_map = dict(zip(csv_paths, pool.map(_read_csv_columns, csv_paths)))
    combined_columns: list[str] = []
    for columns in column_map.values():
        for column in columns:
            if column not in combined_columns:
                combined_columns.append(column)

    # Include variables defined in metadata but absent from all CSVs.
    if column_types:
        for col in column_types:
            if col not in combined_columns:
                combined_columns.append(col)

    if not combined_columns:
        raise ValueError(
            "Unable to determine the column names for the provided CSV files."
        )

    select_statements = []
    params: list[str] = []
    for csv_path in csv_paths:
        csv_columns = column_map[csv_path]

        # Build a per-CSV types= clause.  DuckDB's types= parameter only
        # accepts columns that actually exist in the CSV file, so we must
        # restrict it to the intersection of csv_columns and column_types.
        types_clause = ""
        if column_types:
            type_entries = []
            for col in csv_columns:
                if col in column_types:
                    escaped = col.replace("'", "''")
                    type_entries.append(f"'{escaped}': '{column_types[col]}'")
            if type_entries:
                types_clause = ", types={" + ", ".join(type_entries) + "}"

        select_fields = []
        for column in combined_columns:
            quoted_column = column.replace('"', '""')
            if column in csv_columns:
                select_fields.append(f'"{quoted_column}"')
            else:
                # Use a typed NULL so the column has the correct DuckDB type.
                col_type = (column_types or {}).get(column, "VARCHAR")
                select_fields.append(f'CAST(NULL AS {col_type}) AS "{quoted_column}"')

        select_fields.append(f'{len(params)} AS "{_SOURCE_CSV_COL}"')
        select_statements.append(
            "SELECT "
            + ", ".join(select_fields)
            + f" FROM read_csv_auto(?, HEADER=TRUE{types_clause})"
        )
        params.append(str(csv_path))

    staging_table = f"{table_name}__staging"
    query = " UNION ALL ".join(select_statements)
    conn.execute(f'CREATE TEMP TABLE "{staging_table}" AS {query}', params)
    try:
        conn.execute(
            f'CREATE TABLE "{table_name}" AS '
            f'SELECT * EXCLUDE ("{_SOURCE_CSV_COL}") FROM "{staging_table}"'
        )
        dataset_csv_map = _datasets_from_staging_table(
            conn, staging_table, csv_paths, column_map
        )
        return PrimaryDataTable(dataset_csv_map=dataset_csv_map, csv_columns=column_map)
    finally:
        conn.execute(f'DROP TABLE IF EXISTS "{staging_table}"')


def _read_csv_columns(csv_path: Path) -> list[str]:
    # Use DuckDB's sniffer to automatically detect delimiters (e.g. semicolon)
    # limit=0 reads only the header/schema without scanning the whole file
    try:
        # Own connection: the module-level default connection is not thread-safe.
        with duckdb.connect() as conn:
            return conn.read_csv(str(csv_path), header=True).columns
    except Exception as e:
        LOGGER.warning("Failed to read the header of CSV file %s: %s", csv_path, e)
        return []


def _read_data_model_csvs(data_model_dir: Path) -> list[Path]:
    return sorted(
        csv_path for csv_path in data_model_dir.glob("*.csv") if csv_path.is_file()
    )


def _datasets_from_staging_table(
    conn: duckdb.DuckDBPyConnection,
    staging_table: str,
    csv_paths: list[Path],
    column_map: dict[Path, list[str]],
) -> dict[str, Path]:
    codes_per_csv: dict[int, set[str]] = {idx: set() for idx in range(len(csv_paths))}
    if any(DATASET_COL in columns for columns in column_map.values()):
        rows = conn.execute(
            f'SELECT DISTINCT "{DATASET_COL}", "{_SOURCE_CSV_COL}" '
            f'FROM "{staging_table}"'
        ).fetchall()
        for dataset_code, source_idx in rows:
            # Filter out empty/None values and strip strings
            if dataset_code:
                codes_per_csv[source_idx].add(str(dataset_code).strip())

    datasets: dict[str, Path] = {}
    for idx, csv_path in enumerate(csv_paths):
        if DATASET_COL not in column_map[csv_path]:
            LOGGER.warning(
                "CSV file %s is missing a 'dataset' column. Skipping it when deriving assigned datasets.",
                csv_path,
            )
            continue
        if not codes_per_csv[idx]:
            LOGGER.warning(
                "CSV file %s does not contain any non-empty 'dataset' values.",
                csv_path,
            )
            continue
        for dataset_code in codes_per_csv[idx]:
            if dataset_code in datasets and datasets[dataset_code] != csv_path:
                raise ValueError(
                    f"Dataset '{dataset_code}' is defined in multiple CSV files inside {csv_path.parent}"
                )
            datasets[dataset_code] = csv_path
    return datasets


def load_all_csvs_from_data_folder(request_id: str) -> str:
    db_path = worker_config.duckdb.path
    folder_path = Path(worker_config.data_path).expanduser()
    if not folder_path.exists():
        raise FileNotFoundError(f"Data folder '{folder_path}' does not exist")

    LOGGER.info(
        "[%s] Loading data folder %s into DuckDB %s",
        request_id,
        folder_path,
        db_path,
    )
    conn = duckdb.connect(str(db_path), read_only=False)
    try:
        _reset_metadata_tables(conn)
        data_model_id = 0
        dataset_id = 0

        candidate_dirs = [folder_path]
        candidate_dirs.extend(
            sorted(path for path in folder_path.glob("*/") if path.is_dir())
        )

        for data_model_dir in candidate_dirs:
            meta_path = Path(data_model_dir) / "CDEsMetadata.json"
            if not meta_path.exists():
                continue

            with open(meta_path) as fp:
                metadata = json.load(fp)

            code = metadata.get("code")
            version = metadata.get("version")
            if not code or not version:
                continue

            csv_paths = _read_data_model_csvs(data_model_dir)
            if not csv_paths:
                continue

            data_model_id += 1
            table_prefix = _sanitize_name(f"{code}:{version}")

            primary_data = _create_primary_data_table(
                conn, table_prefix, csv_paths, metadata
            )
            dataset_csv_map = primary_data.dataset_csv_map
            csv_columns = primary_data.csv_columns
            _load_variables_metadata(conn, table_prefix, metadata)
            properties = {}
            properties["cdes"] = metadata
            conn.execute(
                "INSERT INTO data_models VALUES (?, ?, ?, ?, ?, ?)",
                [
                    data_model_id,
                    code,
                    version,
                    metadata.get("label", code),
                    "ENABLED",
                    json.dumps(properties),
                ],
            )

            assigned_datasets = sorted(dataset_csv_map.keys())
            cde_codes = set(_build_column_types(metadata))

            for dataset_code in assigned_datasets:
                dataset_id += 1
                dataset_csv_path = dataset_csv_map[dataset_code]
                if not csv_columns[dataset_csv_path]:
                    LOGGER.warning(
                        "Could not read the header of CSV file %s for dataset '%s'. "
                        "Its variables will be reported as unknown.",
                        dataset_csv_path,
                        dataset_code,
                    )
                dataset_properties = _build_dataset_properties(
                    csv_columns[dataset_csv_path], cde_codes
                )
                conn.execute(
                    "INSERT INTO datasets VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [
                        dataset_id,
                        data_model_id,
                        dataset_code,
                        dataset_code,
                        "ENABLED",
                        str(dataset_csv_path),
                        json.dumps(dataset_properties),
                    ],
                )
    finally:
        conn.close()

    LOGGER.info("[%s] Data folder %s loaded", request_id, folder_path)
    return "Folder import finished"
