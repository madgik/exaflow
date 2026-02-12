import csv
from pathlib import Path

import duckdb
import pytest

from exaflow.worker.utils.duck_db_csv_loader import _create_primary_data_table
from exaflow.worker.utils.duck_db_csv_loader import _read_csv_columns


@pytest.fixture
def semicolon_csv(tmp_path):
    csv_path = tmp_path / "semicolon.csv"
    with csv_path.open("w") as f:
        f.write("col1;col2;col3\n")
        f.write("1;val2;val3\n")
        f.write("2;val2;val3\n")
    return csv_path


def test_read_csv_columns_semicolon(semicolon_csv):
    """Verifies that _read_csv_columns correctly identifies columns from a semicolon-separated file."""
    columns = _read_csv_columns(semicolon_csv)
    # The current implementation fails this, returning ['col1;col2;col3']
    assert columns == ["col1", "col2", "col3"]


def test_create_table_semicolon(semicolon_csv):
    """Verifies that the full table creation works with semicolon CSVs."""
    with duckdb.connect(":memory:") as conn:
        _create_primary_data_table(conn, "test", [semicolon_csv])

        # Verify data
        res = conn.execute('SELECT * FROM "test__primary_data"').fetchall()
        assert len(res) == 2
        assert res[0] == (1, "val2", "val3")
