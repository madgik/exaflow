from pathlib import Path

import pytest

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
