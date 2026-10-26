import json
import warnings
from pathlib import Path
from typing import List

from pydantic import ValidationError

from exaflow.worker import config as worker_config
from exaflow.worker.utils.logger import init_logger
from exaflow.worker.worker_info import duckdb_adapter as metadata_db
from exaflow.worker_communication import CommonDataElement
from exaflow.worker_communication import CommonDataElements
from exaflow.worker_communication import DataModelAttributes
from exaflow.worker_communication import DatasetInfo
from exaflow.worker_communication import DatasetProperties

HEALTHCHECK_VALIDATION_STRING = "HEALTHCHECK"
LOGGER = init_logger("WORKER INFO DB")


def get_data_models() -> List[str]:
    """
    Retrieves the enabled data_models from the database.

    Returns
    ------
    List[str]
        The data_models.
    """

    data_models_code_and_version = metadata_db.execute_and_fetchall(
        f"""SELECT code, version
            FROM data_models
            WHERE status = 'ENABLED'
        """
    )
    data_models = [
        code + ":" + version for code, version in data_models_code_and_version
    ]
    return data_models


def get_dataset_infos(data_model: str) -> List[DatasetInfo]:
    """
    Retrieves the enabled dataset, for a specific data_model.

    Returns
    ------
    Dict[str, str]
        The datasets.
    """
    data_model_code, data_model_version = data_model.split(":")

    datasets_rows = metadata_db.execute_and_fetchall(
        f"""
        SELECT code, label, properties
        FROM datasets
        WHERE data_model_id =
        (
            SELECT data_model_id
            FROM data_models
            WHERE code = '{data_model_code}'
            AND version = '{data_model_version}'
        )
        AND status = 'ENABLED'
        """
    )
    return [
        DatasetInfo(
            code=row[0],
            label=row[1],
            variables=_extract_variables_from_properties(row[2]),
        )
        for row in datasets_rows
    ]


def _extract_variables_from_properties(properties_json) -> List[str]:
    if not properties_json:
        return []

    try:
        dataset_properties = DatasetProperties.model_validate_json(properties_json)
    except (TypeError, ValidationError):
        warnings.warn(
            "Failed to parse dataset properties JSON; returning empty variables list.",
            RuntimeWarning,
        )
        return []

    return dataset_properties.variables


def convert_csv_paths_to_absolute(dataset_path: str) -> str:
    """Anchor a stored CSV path at the configured data directory.

    ``~`` in the data directory or the stored path is expanded, and the
    result is resolved. An absolute path already inside that directory is
    kept. A relative path is joined onto it. ``pathlib`` decides whether
    the stored path is inside the data directory, so a directory such as
    ``/data`` does not match the same text inside ``data.csv``. A trailing
    slash on the data directory does not change the result.

    Example:
        Assuming ``worker_config.data_path`` is "/opt/csvs":

        >>> convert_csv_paths_to_absolute("/opt/csvs/my_folder/data.csv")
        '/opt/csvs/my_folder/data.csv'

        >>> convert_csv_paths_to_absolute("my_folder/data.csv")
        '/opt/csvs/my_folder/data.csv'
    """
    base = Path(worker_config.data_path).expanduser().resolve()
    path = Path(dataset_path).expanduser()
    if path.is_absolute() and path.resolve().is_relative_to(base):
        return str(path.resolve())
    return str((base / path).resolve())


def get_dataset_csv_paths(data_model, datasets: List[str]) -> List[str]:
    """
    Retrieves the enabled dataset csv_paths.
    """
    data_model_code, data_model_version = data_model.split(":")
    datasets_rows = metadata_db.execute_and_fetchall(
        f"""
        SELECT csv_path
        FROM datasets
        WHERE data_model_id =
        (
            SELECT data_model_id
            FROM data_models
            WHERE code = '{data_model_code}'
            AND version = '{data_model_version}'
        )
        AND code IN ({", ".join("'" + str(value) + "'" for value in datasets)})
        AND status = 'ENABLED'
        """
    )
    return [
        convert_csv_paths_to_absolute(row[0]) if row[0] is not None else None
        for row in datasets_rows
    ]


def get_data_model_cdes(data_model: str) -> CommonDataElements:
    """
    Retrieves the cdes of the specific data_model.

    Returns
    ------
    CommonDataElements
        A CommonDataElements object
    """
    data_model_code, data_model_version = data_model.split(":")
    sanitized = _sanitize_table_name(f"{data_model_code}:{data_model_version}")
    table_name = f"{sanitized}_variables_metadata"
    query = f'SELECT code, metadata FROM "{table_name}"'
    cdes_rows = metadata_db.execute_and_fetchall(query)
    cdes = CommonDataElements(
        values={
            code: CommonDataElement.model_validate_json(metadata)
            for code, metadata in cdes_rows
        }
    )

    return cdes


def get_data_model_attributes(data_model: str) -> DataModelAttributes:
    """
    Retrieves the attributes, for a specific data_model.

    Returns
    ------
    DataModelAttributes
    """
    data_model_code, data_model_version = data_model.split(":")

    rows = metadata_db.execute_and_fetchall(
        f"SELECT properties FROM data_models WHERE code = '{data_model_code}' AND version = '{data_model_version}'"
    )
    raw = json.loads(rows[0][0]) if rows else {}
    return DataModelAttributes(
        tags=raw.get("tags", []),
        properties=raw.get("properties", raw),
    )


def check_database_connection():
    """
    Check that the connection with the database is working.
    """
    result = metadata_db.execute_and_fetchall(
        f"SELECT '{HEALTHCHECK_VALIDATION_STRING}'"
    )
    assert result[0][0] == HEALTHCHECK_VALIDATION_STRING


def _sanitize_table_name(name: str) -> str:
    sanitized = name
    for ch in (":", "-", "."):
        sanitized = sanitized.replace(ch, "_")
    return sanitized
