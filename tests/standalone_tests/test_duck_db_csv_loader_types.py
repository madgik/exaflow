import csv
import tempfile
from pathlib import Path

import duckdb
import pytest

from exaflow.worker.utils.duck_db_csv_loader import _build_column_types
from exaflow.worker.utils.duck_db_csv_loader import _create_primary_data_table


@pytest.fixture()
def tmp_csv(tmp_path):
    """Create a CSV where categorical column has integer-like values."""
    csv_path = tmp_path / "data.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "category", "age", "score"])
        writer.writerow(["ds1", "0", "25", "1.5"])
        writer.writerow(["ds1", "1", "30", "2.0"])
        writer.writerow(["ds1", None, "35", "3.7"])
    return csv_path


METADATA = {
    "variables": [
        {
            "code": "dataset",
            "isCategorical": True,
            "sql_type": "text",
        },
        {
            "code": "category",
            "isCategorical": True,
            "sql_type": "text",
        },
    ],
    "groups": [
        {
            "variables": [
                {
                    "code": "age",
                    "isCategorical": False,
                    "sql_type": "int",
                },
                {
                    "code": "score",
                    "isCategorical": False,
                    "sql_type": "real",
                },
                {
                    "code": "diagnosis",
                    "isCategorical": True,
                    "sql_type": "text",
                },
            ]
        }
    ],
}


def test_build_column_types_maps_categorical_to_varchar():
    col_types = _build_column_types(METADATA)
    assert col_types["dataset"] == "VARCHAR"
    assert col_types["category"] == "VARCHAR"
    assert col_types["age"] == "INTEGER"
    assert col_types["score"] == "DOUBLE"
    assert col_types["diagnosis"] == "VARCHAR"


def test_create_primary_data_table_enforces_types(tmp_csv):
    with duckdb.connect(":memory:") as conn:
        _create_primary_data_table(conn, "test", [tmp_csv], METADATA)

        # Check column types
        schema = conn.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = 'test__primary_data' ORDER BY ordinal_position"
        ).fetchall()

        type_map = {name: dtype for name, dtype in schema}
        assert type_map["dataset"] == "VARCHAR"
        assert type_map["category"] == "VARCHAR"
        assert type_map["age"] == "INTEGER"
        assert type_map["score"] == "DOUBLE"
        # Column from metadata but not in CSV should still exist
        assert type_map["diagnosis"] == "VARCHAR"

        # Verify categorical values are strings, not integers
        values = conn.execute(
            'SELECT "category" FROM "test__primary_data" ORDER BY "category"'
        ).fetchall()
        assert len(values) == 3
        assert values == [("0",), ("1",), (None,)]


def test_metadata_only_columns_are_null(tmp_csv):
    """Columns defined in metadata but missing from CSVs should appear as NULL."""

    with duckdb.connect(":memory:") as conn:
        _create_primary_data_table(conn, "test", [tmp_csv], METADATA)

        values = conn.execute('SELECT "diagnosis" FROM "test__primary_data"').fetchall()
        assert len(values) == 3
        assert all(row[0] is None for row in values)


def test_create_primary_data_table_without_types_still_works(tmp_csv):
    """Ensure backward compatibility when no column_types are passed."""
    with duckdb.connect(":memory:") as conn:
        _create_primary_data_table(conn, "test", [tmp_csv])

        count = conn.execute('SELECT COUNT(*) FROM "test__primary_data"').fetchone()[0]
        assert count == 3
