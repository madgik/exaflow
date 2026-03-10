from exaflow import DType
from exaflow.worker_communication import ColumnDataFloat
from exaflow.worker_communication import ColumnDataInt
from exaflow.worker_communication import ColumnDataStr
from exaflow.worker_communication import TabularDataResult


def test_tabular_data_result():
    columns = [
        ColumnDataFloat(name="float", data=[1.0, 10.0]),
        ColumnDataStr(name="str", data=["2", "20"]),
        ColumnDataInt(name="int", data=[3, 30]),
    ]
    result = TabularDataResult(title="The Title", columns=columns)
    expected_dict = {
        "title": "The Title",
        "columns": [
            {"name": "float", "type": DType.FLOAT, "data": [1.0, 10.0]},
            {"name": "str", "type": DType.STR, "data": ["2", "20"]},
            {"name": "int", "type": DType.INT, "data": [3, 30]},
        ],
    }

    assert result.model_dump() == expected_dict


def test_tabular_data_result_to_json():
    columns = [
        ColumnDataFloat(name="float", data=[1.0, 10.0]),
        ColumnDataStr(name="str", data=["2", "20"]),
        ColumnDataInt(name="int", data=[3, 30]),
    ]
    result = TabularDataResult(title="The Title", columns=columns)
    assert TabularDataResult.model_validate_json(result.model_dump_json()) == result
