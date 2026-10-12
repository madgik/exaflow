import pandas as pd

from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.algorithms.utils.inputdata_utils import _apply_filter
from exaflow.algorithms.utils.inputdata_utils import fetch_data


def _rule(column, operator, value=None):
    return {"id": column, "operator": operator, "value": value}


def test_or_filter_keeps_identical_rows_and_order():
    df = pd.DataFrame({"age": [70, 70, 30, 80], "sex": ["F", "F", "M", "M"]})
    rule = {
        "condition": "OR",
        "rules": [_rule("age", "greater", 75), _rule("sex", "equal", "F")],
    }

    result = _apply_filter(df, rule)

    assert result.index.tolist() == [0, 1, 3]


def test_nested_and_with_not_in():
    df = pd.DataFrame({"age": [70, 60, 30, 80], "sex": ["F", "M", "M", "X"]})
    rule = {
        "condition": "AND",
        "rules": [
            _rule("age", "greater_or_equal", 60),
            {"condition": "OR", "rules": [_rule("sex", "not_in", ["X", "M"])]},
        ],
    }

    assert _apply_filter(df, rule).index.tolist() == [0]


def test_fetch_data_reads_filter_columns_that_are_not_variables(tmp_path):
    csv_path = tmp_path / "data.csv"
    pd.DataFrame(
        {"dataset": ["d1", "d1", "d1"], "age": [50, 70, 80], "score": [1.0, 2.0, 3.0]}
    ).to_csv(csv_path, index=False)
    inputdata = Inputdata(
        data_model="dm:1",
        datasets=["d1"],
        variables=["score"],
        filters={"condition": "AND", "rules": [_rule("age", "greater", 60)]},
    )

    result = fetch_data(inputdata, [str(csv_path)])

    assert result["score"].tolist() == [2.0, 3.0]
