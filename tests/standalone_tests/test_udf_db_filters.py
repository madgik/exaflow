import duckdb
import pytest

from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.data_filters import FilterError
from exaflow.worker import config as worker_config
from exaflow.worker.exareme3.udf.udf_db import _fetch_with_duckdb
from exaflow.worker.exareme3.udf.udf_db import primary_table_name
from exaflow.worker.exareme3.udf.udf_db import quote_identifier

DATA_MODEL = "cohort:1.0"
INJECTION_FILTER = {
    "id": "s",
    "operator": "equal",
    "value": ['][1]) OR (1=1) --"'],
}


def _expect_filter_error(callback):
    try:
        result = callback()
    except FilterError:
        return
    except Exception as exc:
        pytest.fail(f"expected FilterError, got {type(exc).__name__}: {exc}")
    datasets = result.column("dataset").to_pylist()
    pytest.fail(
        "filter returned rows instead of raising FilterError; "
        f"datasets={datasets} rows={result.to_pydict()}"
    )


@pytest.fixture
def two_dataset_db(tmp_path, monkeypatch):
    db_path = tmp_path / "worker.duckdb"
    table_name = primary_table_name(DATA_MODEL)
    with duckdb.connect(str(db_path)) as conn:
        conn.execute(
            f"CREATE TABLE {table_name} (dataset VARCHAR, s VARCHAR, age INTEGER)"
        )
        conn.execute(
            f"INSERT INTO {table_name} VALUES "
            "('alpha', 'keep', 40), "
            "('alpha', 'O''Hara', 10), "
            "('beta', 'other', 40)"
        )
    monkeypatch.setattr(worker_config.duckdb, "path", str(db_path))
    return db_path


def _inputdata(datasets, filters):
    return Inputdata(
        data_model=DATA_MODEL,
        datasets=datasets,
        filters=filters,
        variables=["s", "age"],
    )


def _fetch(inputdata):
    return _fetch_with_duckdb(inputdata, {"dataset", "s", "age"})


def test_primary_table_name_escapes_double_quotes():
    quoted = primary_table_name('a:b"c')
    assert quoted == quote_identifier('a_b"c__primary_data')
    with duckdb.connect() as conn:
        conn.execute(f"CREATE TABLE {quoted} (dataset VARCHAR)")
        conn.execute(f"INSERT INTO {quoted} VALUES ('alpha')")
        assert conn.execute(f"SELECT dataset FROM {quoted}").fetchall() == [("alpha",)]


def test_primary_table_name_sanitizes_separators():
    assert primary_table_name("a:b-c.d") == '"a_b_c_d__primary_data"'


def test_fetch_with_duckdb_keeps_rows_inside_the_requested_dataset(two_dataset_db):
    kept = _fetch(
        _inputdata(["alpha"], {"id": "s", "operator": "equal", "value": "keep"})
    )
    assert kept.column("dataset").to_pylist() == ["alpha"]
    assert kept.column("s").to_pylist() == ["keep"]

    named = _fetch(
        _inputdata(["alpha"], {"id": "s", "operator": "equal", "value": "O'Hara"})
    )
    assert named.column("dataset").to_pylist() == ["alpha"]
    assert named.column("age").to_pylist() == [10]

    older = _fetch(
        _inputdata(["alpha"], {"id": "age", "operator": "greater", "value": 30})
    )
    assert older.column("dataset").to_pylist() == ["alpha"]
    assert older.column("s").to_pylist() == ["keep"]


def test_fetch_with_duckdb_injection_payload_cannot_reach_other_dataset(
    two_dataset_db,
):
    _expect_filter_error(_fetch_injection)

    with duckdb.connect(str(two_dataset_db), read_only=True) as conn:
        beta = conn.execute(
            f"SELECT dataset FROM {primary_table_name(DATA_MODEL)} "
            "WHERE dataset = 'beta'"
        ).fetchall()
    assert beta == [("beta",)]


def _fetch_injection():
    return _fetch(_inputdata(["alpha"], INJECTION_FILTER))


@pytest.mark.parametrize(
    "filters",
    [
        INJECTION_FILTER,
        {"id": "s", "operator": "between", "value": "abc"},
        {"id": "age", "operator": "between", "value": [1]},
        {"id": "age", "operator": "between", "value": 5},
        {"id": "age", "operator": "not_between", "value": [1, 2, 3]},
        {"id": "s", "operator": "in", "value": []},
        {"id": "s", "operator": "not_in", "value": "abc"},
        {"id": "age", "operator": "equal", "value": None},
        {"id": "s", "operator": "not_in", "value": ["keep", None]},
        {"id": "s", "operator": "equal", "value": "a\x00b"},
    ],
)
def test_fetch_with_duckdb_malformed_filters_raise_filter_error(
    two_dataset_db, filters
):
    _expect_filter_error(lambda: _fetch(_inputdata(["alpha"], filters)))
