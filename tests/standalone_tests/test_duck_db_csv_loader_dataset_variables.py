import csv
import json
from unittest.mock import MagicMock

import duckdb
import pytest

from exaflow.worker import config as worker_config
from exaflow.worker.utils import duck_db_csv_loader
from exaflow.worker.utils.duck_db_csv_loader import _build_dataset_properties
from exaflow.worker.utils.duck_db_csv_loader import _read_csv_columns
from exaflow.worker.utils.duck_db_csv_loader import load_all_csvs_from_data_folder
from exaflow.worker.worker_info import duckdb_adapter
from exaflow.worker.worker_info.worker_info_db import get_dataset_infos

METADATA = {
    "code": "testmodel",
    "version": "0.1",
    "label": "Test model",
    "variables": [
        {
            "code": "dataset",
            "type": "nominal",
            "enumerations": [
                {"code": "full", "label": "full"},
                {"code": "partial_a", "label": "partial_a"},
                {"code": "partial_b", "label": "partial_b"},
            ],
        },
        {"code": "gender", "type": "nominal"},
    ],
    "groups": [
        {
            "code": "measurements",
            "variables": [
                {"code": "age", "type": "discrete"},
                {"code": "score", "type": "continuous"},
            ],
        }
    ],
}


def _write_csv(path, header, rows):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


@pytest.fixture
def data_folder(tmp_path, monkeypatch):
    data_path = tmp_path / "data"
    model_dir = data_path / "testmodel_v_0_1"
    model_dir.mkdir(parents=True)
    (model_dir / "CDEsMetadata.json").write_text(json.dumps(METADATA))

    # Has every CDE.
    _write_csv(
        model_dir / "full.csv",
        ["dataset", "gender", "age", "score"],
        [["full", "F", "30", "1.5"]],
    )
    # Lacks "age" and has a column ("row_id") that is not a CDE.
    # Two datasets share this CSV, so both get the same variables.
    _write_csv(
        model_dir / "partial.csv",
        ["row_id", "score", "dataset", "gender"],
        [["1", "2.0", "partial_a", "M"], ["2", "3.0", "partial_b", "F"]],
    )

    duckdb_path = tmp_path / "db.duckdb"
    monkeypatch.setattr(worker_config, "data_path", str(data_path))
    monkeypatch.setattr(worker_config.duckdb, "path", str(duckdb_path))
    monkeypatch.setattr(duckdb_adapter, "_duckdb_path", duckdb_path)
    return model_dir, duckdb_path


@pytest.fixture
def loaded_data_folder(data_folder):
    load_all_csvs_from_data_folder("test-request")
    return data_folder[1]


def _stored_properties(duckdb_path):
    with duckdb.connect(str(duckdb_path), read_only=True) as conn:
        rows = conn.execute("SELECT code, properties FROM datasets").fetchall()
    return {code: json.loads(properties) for code, properties in rows}


def test_build_dataset_properties_keeps_header_order_and_drops_non_cdes():
    properties = _build_dataset_properties(
        ["row_id", "score", "dataset", "gender"], {"dataset", "gender", "age", "score"}
    )
    assert properties == {
        "tags": [],
        "properties": {"variables": ["score", "dataset", "gender"]},
    }


def test_loader_fills_dataset_properties_from_csv_header(loaded_data_folder):
    stored = _stored_properties(loaded_data_folder)

    assert stored["full"] == {
        "tags": [],
        "properties": {"variables": ["dataset", "gender", "age", "score"]},
    }


def test_loader_lists_only_present_cdes_for_partial_dataset(loaded_data_folder):
    stored = _stored_properties(loaded_data_folder)

    expected = {"tags": [], "properties": {"variables": ["score", "dataset", "gender"]}}
    assert stored["partial_a"] == expected
    assert stored["partial_b"] == expected


def test_dataset_infos_expose_loaded_variables(loaded_data_folder):
    infos = {info.code: info for info in get_dataset_infos("testmodel:0.1")}

    assert infos["full"].variables == ["dataset", "gender", "age", "score"]
    assert infos["partial_a"].variables == ["score", "dataset", "gender"]
    assert infos["partial_b"].variables == ["score", "dataset", "gender"]


def test_read_csv_columns_warns_with_path_when_header_unreadable(tmp_path, monkeypatch):
    warning = MagicMock()
    monkeypatch.setattr(duck_db_csv_loader.LOGGER, "warning", warning)
    missing_csv = tmp_path / "missing.csv"

    assert _read_csv_columns(missing_csv) == []

    warning.assert_called_once()
    assert missing_csv in warning.call_args.args


def test_loader_warns_with_dataset_and_path_when_header_unreadable(
    data_folder, monkeypatch
):
    model_dir, duckdb_path = data_folder
    partial_csv = model_dir / "partial.csv"
    real_create_table = duck_db_csv_loader._create_primary_data_table

    def create_table_with_unreadable_partial_header(*args, **kwargs):
        result = real_create_table(*args, **kwargs)
        return result._replace(csv_columns={**result.csv_columns, partial_csv: []})

    monkeypatch.setattr(
        duck_db_csv_loader,
        "_create_primary_data_table",
        create_table_with_unreadable_partial_header,
    )
    warning = MagicMock()
    monkeypatch.setattr(duck_db_csv_loader.LOGGER, "warning", warning)

    load_all_csvs_from_data_folder("test-request")

    warned_args = [call.args for call in warning.call_args_list]
    for dataset_code in ("partial_a", "partial_b"):
        assert any(partial_csv in args and dataset_code in args for args in warned_args)
    assert not any("full" in args for args in warned_args)
    stored = _stored_properties(duckdb_path)
    assert stored["partial_a"]["properties"]["variables"] == []
    assert stored["full"]["properties"]["variables"] == [
        "dataset",
        "gender",
        "age",
        "score",
    ]
