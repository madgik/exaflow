import json
from pathlib import Path

import duckdb
import pytest

from exaflow.worker.utils.duck_db_csv_loader import _reset_metadata_tables
from exaflow.worker.worker_info import duckdb_adapter
from exaflow.worker.worker_info import worker_info_db


@pytest.mark.parametrize(
    "data_path, stored_path, expected",
    [
        (
            "/data/datasets",
            "/data/datasets/dementia/data.csv",
            "/data/datasets/dementia/data.csv",
        ),
        (
            "/data/datasets",
            "dementia/data.csv",
            "/data/datasets/dementia/data.csv",
        ),
        (
            "/opt/csvs",
            "/opt/csvs/dementia/data.csv",
            "/opt/csvs/dementia/data.csv",
        ),
        (
            "/opt/csvs",
            "dementia/data.csv",
            "/opt/csvs/dementia/data.csv",
        ),
    ],
)
def test_convert_csv_paths_to_absolute(monkeypatch, data_path, stored_path, expected):
    monkeypatch.setattr(worker_info_db.worker_config, "data_path", data_path)

    assert worker_info_db.convert_csv_paths_to_absolute(stored_path) == expected


def test_convert_csv_paths_to_absolute_when_data_path_is_prefix_of_filename(
    monkeypatch,
):
    """'/data' must not match the '/data' inside 'data.csv'."""
    monkeypatch.setattr(worker_info_db.worker_config, "data_path", "/data")

    assert (
        worker_info_db.convert_csv_paths_to_absolute("/data/dementia/data.csv")
        == "/data/dementia/data.csv"
    )
    assert (
        worker_info_db.convert_csv_paths_to_absolute("dementia/data.csv")
        == "/data/dementia/data.csv"
    )


def test_convert_csv_paths_to_absolute_trailing_slash(monkeypatch):
    monkeypatch.setattr(worker_info_db.worker_config, "data_path", "/data/")

    assert (
        worker_info_db.convert_csv_paths_to_absolute("/data/dementia/data.csv")
        == "/data/dementia/data.csv"
    )
    assert (
        worker_info_db.convert_csv_paths_to_absolute("dementia/data.csv")
        == "/data/dementia/data.csv"
    )
    # The data directory string can also appear later in a relative path.
    assert (
        worker_info_db.convert_csv_paths_to_absolute("study/data/measurements.csv")
        == "/data/study/data/measurements.csv"
    )


def test_convert_csv_paths_to_absolute_expands_user(monkeypatch):
    monkeypatch.setattr(worker_info_db.worker_config, "data_path", "~/datasets")
    expected = str(Path("~/datasets").expanduser() / "dementia" / "data.csv")

    assert worker_info_db.convert_csv_paths_to_absolute("dementia/data.csv") == expected
    stored = str(Path("~/datasets").expanduser() / "dementia" / "data.csv")
    assert worker_info_db.convert_csv_paths_to_absolute(stored) == expected


# A quote in the data model code used to break the string-built SQL.
DATA_MODEL_CODE = "o'brien"
DATA_MODEL = f"{DATA_MODEL_CODE}:1.0"


@pytest.fixture
def metadata_db(tmp_path, monkeypatch):
    db_path = tmp_path / "worker.duckdb"
    with duckdb.connect(str(db_path)) as conn:
        _reset_metadata_tables(conn)
        conn.execute(
            "INSERT INTO data_models VALUES (1, ?, '1.0', 'label', 'ENABLED', ?)",
            [DATA_MODEL_CODE, json.dumps({"tags": ["longitudinal"]})],
        )
        conn.executemany(
            "INSERT INTO datasets VALUES (?, 1, ?, ?, 'ENABLED', ?, NULL)",
            [
                [1, "ds1", "ds1", "/data/ds1.csv"],
                [2, "ds'2", "ds'2", "/data/ds2.csv"],
            ],
        )
    monkeypatch.setattr(duckdb_adapter, "_duckdb_path", db_path)
    monkeypatch.setattr(worker_info_db.worker_config, "data_path", "/data")
    return db_path


def test_dataset_infos_handle_quotes_in_identifiers(metadata_db):
    infos = worker_info_db.get_dataset_infos(DATA_MODEL)

    assert sorted(info.code for info in infos) == ["ds'2", "ds1"]


def test_dataset_csv_paths_handle_quotes_in_identifiers(metadata_db):
    paths = worker_info_db.get_dataset_csv_paths(DATA_MODEL, ["ds'2"])

    assert len(paths) == 1 and paths[0].endswith("/ds2.csv")
    assert worker_info_db.get_dataset_csv_paths(DATA_MODEL, []) == []


def test_data_model_attributes_handle_quotes_in_identifiers(metadata_db):
    attributes = worker_info_db.get_data_model_attributes(DATA_MODEL)

    assert attributes.tags == ["longitudinal"]


def test_check_database_connection(metadata_db):
    worker_info_db.check_database_connection()
