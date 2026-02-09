import json
import warnings
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
LOGGER = init_logger("WORKER DATA LOADER")


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
        dataset_properties = DatasetProperties.parse_raw(properties_json)
    except (TypeError, ValidationError):
        warnings.warn(
            "Failed to parse dataset properties JSON; returning empty variables list.",
            RuntimeWarning,
        )
        return []

    return dataset_properties.variables


def convert_csv_paths_to_absolute(dataset_path: str) -> str:
    """
    Convert a CSV file path to an absolute path using the configured data directory.

    This function addresses the scenario where a CSV file's path stored in the database may be relative,
    rather than absolute. By leveraging the base directory defined in `worker_config.data_path`, it constructs
    a complete, absolute path for the CSV file. The function does this by splitting the provided `dataset_path`
    by the base directory string, then taking the last part (which represents the relative path) and concatenating
    it with the base directory.

    Parameters:
        dataset_path (str): The CSV file path as imported by the database. This path may be relative or partially
                            include the base directory.

    Returns:
        str: The absolute path to the CSV file, formed by combining `worker_config.data_path` and the relative portion
             of `dataset_path`.

    Example:
        Assuming `worker_config.data_path` is "/data/datasets", then:

        >>> convert_csv_paths_to_absolute("/data/datasets/my_folder/data.csv")
        '/data/datasets/my_folder/data.csv'

        >>> convert_csv_paths_to_absolute("my_folder/data.csv")
        '/data/datasets/my_folder/data.csv'

    Note:
        The function splits the original path by the base directory string. The resulting list's last element is treated
        as the relative path. This ensures that even if the CSV path is stored in a relative format, it is transformed
        into an absolute path using the current configuration.
    """
    relative_path = dataset_path.split(str(worker_config.data_path))[-1]
    return f"{worker_config.data_path}/{relative_path}"


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
        AND code IN ({', '.join("'" + str(value) + "'" for value in datasets)})
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
            code: CommonDataElement.parse_raw(metadata) for code, metadata in cdes_rows
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
