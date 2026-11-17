import csv
import tempfile
from pathlib import Path

import duckdb
import pytest

from exaflow.worker.utils.duck_db_csv_loader import _build_column_types
from exaflow.worker.utils.duck_db_csv_loader import _create_primary_data_table


@pytest.fixture()
def tmp_csv(tmp_path):
    """Create a CSV whose categorical columns hold integer codes."""
    csv_path = tmp_path / "data.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "category", "age", "score"])
        writer.writerow(["1", "1", "25", "1.5"])
        writer.writerow(["1", "2", "30", "2.0"])
        writer.writerow(["1", None, "35", "3.7"])
    return csv_path


def _enums(*codes):
    return [{"code": code, "label": f"Level {code}"} for code in codes]


METADATA = {
    "variables": [
        {
            "code": "dataset",
            "type": "nominal",
            "enumerations": _enums("1", "2", "3"),
        },
        {
            "code": "category",
            "type": "nominal",
            "enumerations": _enums("1", "2"),
        },
    ],
    "groups": [
        {
            "variables": [
                {
                    "code": "age",
                    "type": "discrete",
                },
                {
                    "code": "score",
                    "type": "continuous",
                },
                {
                    "code": "diagnosis",
                    "type": "nominal",
                    "enumerations": _enums("1", "2"),
                },
            ]
        }
    ],
}


def test_build_column_types_maps_categorical_to_integer():
    col_types = _build_column_types(METADATA)
    assert col_types["dataset"] == "INTEGER"
    assert col_types["category"] == "INTEGER"
    assert col_types["age"] == "INTEGER"
    assert col_types["score"] == "DOUBLE"
    assert col_types["diagnosis"] == "INTEGER"


@pytest.mark.parametrize("code", ["var", "visitid"])
@pytest.mark.parametrize("cde_type", ["nominal", "ordinal", "binary"])
@pytest.mark.parametrize("enumerations", [None, [], _enums("a", "b")])
def test_categorical_without_integer_enumerations_is_rejected(
    code, cde_type, enumerations
):
    variable = {"code": code, "type": cde_type}
    if enumerations is not None:
        variable["enumerations"] = enumerations

    with pytest.raises(ValueError, match=f"'{code}'.*integer codes"):
        _build_column_types({"variables": [variable]})


def test_unsupported_variable_type_has_a_clear_error():
    with pytest.raises(ValueError) as exc:
        _build_column_types({"variables": [{"code": "var", "type": "real"}]})
    assert "real" in str(exc.value)
    assert "categorical CDE" not in str(exc.value)


@pytest.mark.parametrize(
    "variable",
    [
        {"code": "subjectid", "type": "discrete"},
        {
            "code": "visitid",
            "type": "nominal",
            "enumerations": _enums("1", "2"),
        },
    ],
    ids=["discrete-subjectid", "nominal-visitid"],
)
def test_identifier_columns_are_integer(variable):
    assert _build_column_types({"variables": [variable]}) == {
        variable["code"]: "INTEGER"
    }


def test_create_primary_data_table_enforces_types(tmp_csv):
    with duckdb.connect(":memory:") as conn:
        _create_primary_data_table(conn, "test", [tmp_csv], METADATA)

        # Check column types
        schema = conn.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = 'test__primary_data' ORDER BY ordinal_position"
        ).fetchall()

        type_map = {name: dtype for name, dtype in schema}
        assert type_map["dataset"] == "INTEGER"
        assert type_map["category"] == "INTEGER"
        assert type_map["age"] == "INTEGER"
        assert type_map["score"] == "DOUBLE"
        # Column from metadata but not in CSV should still exist
        assert type_map["diagnosis"] == "INTEGER"

        # Verify categorical values are stored as integer codes
        values = conn.execute(
            'SELECT "category" FROM "test__primary_data" ORDER BY "category"'
        ).fetchall()
        assert len(values) == 3
        assert values == [(1,), (2,), (None,)]


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


def _write_csv(path, header, rows):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def test_create_primary_data_table_returns_dataset_csv_map(tmp_path):
    first = _write_csv(tmp_path / "a.csv", ["dataset", "age"], [["1", 1], ["2", 2]])
    second = _write_csv(tmp_path / "b.csv", ["dataset", "age"], [[" 3 ", 3]])
    no_dataset = _write_csv(tmp_path / "c.csv", ["age"], [[4]])

    with duckdb.connect(":memory:") as conn:
        result = _create_primary_data_table(
            conn, "test", [first, second, no_dataset], METADATA
        )
        tables = {
            row[0]
            for row in conn.execute("SELECT table_name FROM duckdb_tables()").fetchall()
        }
        columns = [
            row[0] for row in conn.execute('DESCRIBE "test__primary_data"').fetchall()
        ]

    assert result.dataset_csv_map == {"1": first, "2": first, "3": second}
    assert result.csv_columns == {
        first: ["dataset", "age"],
        second: ["dataset", "age"],
        no_dataset: ["age"],
    }
    assert tables == {"test__primary_data"}
    assert columns[0] == "dataset" and "__source_csv_idx" not in columns


def test_create_primary_data_table_rejects_dataset_in_multiple_csvs(tmp_path):
    first = _write_csv(tmp_path / "a.csv", ["dataset", "age"], [["1", 1]])
    second = _write_csv(tmp_path / "b.csv", ["dataset", "age"], [["1", 2]])

    with duckdb.connect(":memory:") as conn:
        with pytest.raises(ValueError, match="defined in multiple CSV files"):
            _create_primary_data_table(conn, "test", [first, second], METADATA)


@pytest.mark.parametrize(
    "prefix", ['quoted"model', 'x"; CREATE TABLE injected(a INT); --']
)
def test_model_identifiers_cannot_execute_sql(tmp_csv, prefix):
    from exaflow.worker.utils.duck_db_csv_loader import _load_variables_metadata

    with duckdb.connect(":memory:") as conn:
        _create_primary_data_table(conn, prefix, [tmp_csv], METADATA)
        _load_variables_metadata(conn, prefix, METADATA)
        names = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        assert prefix + "__primary_data" in names
        assert prefix + "_variables_metadata" in names
        assert "injected" not in names
