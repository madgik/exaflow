import json

import duckdb
import pytest

from exaflow.worker.utils.duck_db_csv_loader import _reset_metadata_tables
from exaflow.worker.worker_info import duckdb_adapter
from exaflow.worker.worker_info import worker_info_db

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
