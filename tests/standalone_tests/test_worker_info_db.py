import pytest

from exaflow.worker.worker_info import worker_info_db


@pytest.mark.parametrize(
    "stored_path",
    ["/data/datasets/dementia/data.csv", "dementia/data.csv"],
)
def test_convert_csv_paths_to_absolute(monkeypatch, stored_path):
    monkeypatch.setattr(worker_info_db.worker_config, "data_path", "/data/datasets")

    assert (
        worker_info_db.convert_csv_paths_to_absolute(stored_path)
        == "/data/datasets/dementia/data.csv"
    )
