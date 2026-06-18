import pandas as pd
import pytest

from exaflow.algorithms.exareme3.preprocessing.missing_values_handler import (
    MissingValuesHandler,
)
from exaflow.algorithms.exareme3.preprocessing.missing_values_handler import (
    MissingValueStrategy,
)
from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.worker_communication import BadUserInput
from exaflow.worker_communication import InsufficientDataError


def _make_handler(params):
    return MissingValuesHandler(params=params)


def _validate(handler):
    handler.validate_params(
        inputdata=Inputdata(
            data_model="dm:0.1",
            datasets=["d1"],
            variables=["x1", "x2", "x3", "x_cat", "y1"],
        ),
        metadata={
            "x1": {"is_categorical": False},
            "x2": {"is_categorical": False},
            "x3": {"is_categorical": False},
            "x_cat": {
                "is_categorical": True,
                "enumerations": {"A": "Category A", "B": "Category B"},
            },
            "y1": {"is_categorical": False},
        },
    )


def test_get_specification_has_expected_shape():
    spec = MissingValuesHandler.get_specification()

    assert spec.name == "missing_values_handler"
    assert spec.enabled is True
    assert set(spec.parameters.keys()) == {"strategies", "fill_values"}
    assert spec.parameters["strategies"].required is True
    assert spec.parameters["strategies"].dict_values_enums.source == [
        MissingValueStrategy.DROP.value,
        MissingValueStrategy.MEAN.value,
        MissingValueStrategy.MEDIAN.value,
        MissingValueStrategy.MOST_FREQUENT.value,
        MissingValueStrategy.CONSTANT.value,
    ]


def test_validate_params_accepts_per_variable_strategies():
    handler = _make_handler(
        params={
            "strategies": {
                "x1": MissingValueStrategy.MEAN.value,
                "x_cat": MissingValueStrategy.MOST_FREQUENT.value,
                "y1": MissingValueStrategy.CONSTANT.value,
            },
            "fill_values": {"y1": 0},
        }
    )

    _validate(handler)


def test_validate_params_rejects_unknown_strategy():
    handler = _make_handler(params={"strategies": {"x1": "unknown"}})

    with pytest.raises(BadUserInput, match="Invalid per-variable strategy"):
        _validate(handler)


def test_validate_params_rejects_unknown_variable_in_strategies():
    handler = _make_handler(
        params={"strategies": {"unknown_var": MissingValueStrategy.MEAN.value}}
    )

    with pytest.raises(
        BadUserInput, match="variables not present in inputdata.variables"
    ):
        _validate(handler)


def test_validate_params_rejects_mean_for_categorical_variable():
    handler = _make_handler(
        params={
            "strategies": {"x_cat": MissingValueStrategy.MEAN.value},
        }
    )

    with pytest.raises(BadUserInput, match="can only be used for numerical variables"):
        _validate(handler)


def test_validate_params_rejects_fill_values_for_non_constant_variable():
    handler = _make_handler(
        params={
            "strategies": {"x1": MissingValueStrategy.MEAN.value},
            "fill_values": {"x1": 0},
        }
    )

    with pytest.raises(
        BadUserInput, match="can only be provided when strategy is 'constant'"
    ):
        _validate(handler)


@pytest.mark.parametrize("fill_value", [0, False, "", 1.5])
def test_validate_params_accepts_constant_strategy_with_falsy_fill_values(fill_value):
    handler = _make_handler(
        params={
            "strategies": {"x1": MissingValueStrategy.CONSTANT.value},
            "fill_values": {"x1": fill_value},
        }
    )

    _validate(handler)


def test_validate_params_rejects_non_scalar_fill_values():
    handler = _make_handler(
        params={
            "strategies": {"x1": MissingValueStrategy.CONSTANT.value},
            "fill_values": {"x1": {"nested": 1}},
        }
    )

    with pytest.raises(BadUserInput, match="should be a scalar"):
        _validate(handler)


def test_validate_params_rejects_fill_values_outside_categorical_enums():
    handler = _make_handler(
        params={
            "strategies": {"x_cat": MissingValueStrategy.CONSTANT.value},
            "fill_values": {"x_cat": "unknown"},
        }
    )

    with pytest.raises(BadUserInput, match="categorical enum codes"):
        _validate(handler)


def test_validate_params_requires_fill_value_for_categorical_constant():
    handler = _make_handler(
        params={
            "strategies": {"x_cat": MissingValueStrategy.CONSTANT.value},
        }
    )

    with pytest.raises(BadUserInput, match="requires 'fill_values\\[x_cat\\]'"):
        _validate(handler)


def test_validate_params_accepts_fill_value_within_categorical_enums():
    handler = _make_handler(
        params={
            "strategies": {"x_cat": MissingValueStrategy.CONSTANT.value},
            "fill_values": {"x_cat": "A"},
        }
    )

    _validate(handler)


def test_validate_params_rejects_non_string_fill_value_for_categorical_enums():
    handler = _make_handler(
        params={
            "strategies": {"x_cat": MissingValueStrategy.CONSTANT.value},
            "fill_values": {"x_cat": 1},
        }
    )

    with pytest.raises(BadUserInput, match="text categorical enum code"):
        handler.validate_params(
            inputdata=Inputdata(
                data_model="dm:0.1",
                datasets=["d1"],
                variables=["x_cat"],
            ),
            metadata={
                "x_cat": {
                    "is_categorical": True,
                    "enumerations": {"1": "One", "2": "Two"},
                },
            },
        )


def test_transform_variables_returns_identity():
    handler = _make_handler(
        params={"strategies": {"x1": MissingValueStrategy.DROP.value}}
    )

    transformed_variables = handler.transform_variables(
        variables=["x1", "x2", "y1"],
    )

    assert transformed_variables == ["x1", "x2", "y1"]


def test_transform_metadata_returns_copy():
    metadata = {"x1": {"is_categorical": False}}
    handler = _make_handler(
        params={"strategies": {"x1": MissingValueStrategy.DROP.value}}
    )

    transformed = handler.transform_metadata(metadata=metadata)
    transformed["x1"]["is_categorical"] = True

    assert metadata["x1"]["is_categorical"] is False


@pytest.mark.parametrize(
    "strategy",
    [
        MissingValueStrategy.MEAN.value,
        MissingValueStrategy.MEDIAN.value,
    ],
)
def test_transform_metadata_promotes_int_to_real_for_float_imputation(strategy):
    metadata = {
        "x1": {"is_categorical": False, "sql_type": "int"},
        "x2": {"is_categorical": False, "sql_type": "real"},
    }
    handler = _make_handler(params={"strategies": {"x1": strategy}})

    transformed = handler.transform_metadata(metadata=metadata)

    assert transformed["x1"]["sql_type"] == "real"
    assert transformed["x2"]["sql_type"] == "real"
    assert metadata["x1"]["sql_type"] == "int"


def test_transform_metadata_keeps_int_for_most_frequent_imputation():
    metadata = {"x1": {"is_categorical": False, "sql_type": "int"}}
    handler = _make_handler(
        params={"strategies": {"x1": MissingValueStrategy.MOST_FREQUENT.value}}
    )

    transformed = handler.transform_metadata(metadata=metadata)

    assert transformed["x1"]["sql_type"] == "int"


def test_transform_data_and_metadata_promotes_float_constant_fill_to_real():
    data = pd.DataFrame({"x1": [1, None, 3]})
    metadata = {"x1": {"is_categorical": False, "sql_type": "int"}}
    handler = _make_handler(
        params={
            "strategies": {"x1": MissingValueStrategy.CONSTANT.value},
            "fill_values": {"x1": 1.5},
        }
    )

    transformed, transformed_metadata = handler.transform_data_and_metadata(
        data=data,
        metadata=metadata,
    )

    assert transformed["x1"].tolist() == [1.0, 1.5, 3.0]
    assert transformed_metadata["x1"]["sql_type"] == "real"
    assert metadata["x1"]["sql_type"] == "int"


def test_transform_metadata_keeps_int_for_integer_constant_fill():
    metadata = {"x1": {"is_categorical": False, "sql_type": "int"}}
    handler = _make_handler(
        params={
            "strategies": {"x1": MissingValueStrategy.CONSTANT.value},
            "fill_values": {"x1": 0},
        }
    )

    transformed = handler.transform_metadata(metadata=metadata)

    assert transformed["x1"]["sql_type"] == "int"


def test_transform_metadata_keeps_int_for_string_constant_fill():
    metadata = {"x1": {"is_categorical": False, "sql_type": "int"}}
    handler = _make_handler(
        params={
            "strategies": {"x1": MissingValueStrategy.CONSTANT.value},
            "fill_values": {"x1": "unknown"},
        }
    )

    transformed = handler.transform_metadata(metadata=metadata)

    assert transformed["x1"]["sql_type"] == "int"


def test_transform_data_per_variable_drop_only_selected_columns():
    data = pd.DataFrame(
        {
            "x1": [1.0, None, 3.0],
            "x2": [10.0, 20.0, None],
            "x3": [5.0, 6.0, 7.0],
        }
    )
    handler = _make_handler(
        params={
            "strategies": {
                "x1": MissingValueStrategy.DROP.value,
                "x2": MissingValueStrategy.MEAN.value,
            }
        }
    )

    transformed = handler.transform_data(data=data)

    expected = pd.DataFrame(
        {
            "x1": [1.0, 3.0],
            "x2": [10.0, 10.0],
            "x3": [5.0, 7.0],
        },
        index=[0, 2],
    )
    pd.testing.assert_frame_equal(
        transformed,
        expected,
        check_dtype=False,
    )


def test_transform_data_per_variable_mixed_strategies():
    data = pd.DataFrame(
        {
            "x1": [1.0, None, 3.0],
            "x2": [10.0, 10.0, None],
            "x3": ["a", None, "b"],
            "x4": [100.0, None, 300.0],
        }
    )
    handler = _make_handler(
        params={
            "strategies": {
                "x1": MissingValueStrategy.MEAN.value,
                "x2": MissingValueStrategy.MOST_FREQUENT.value,
                "x3": MissingValueStrategy.CONSTANT.value,
            },
            "fill_values": {"x3": "unknown"},
        }
    )

    transformed = handler.transform_data(data=data)

    expected = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0],
            "x2": [10.0, 10.0, 10.0],
            "x3": ["a", "unknown", "b"],
            "x4": [100.0, None, 300.0],
        }
    )
    pd.testing.assert_frame_equal(
        transformed.reset_index(drop=True),
        expected,
        check_dtype=False,
    )


def test_transform_data_per_variable_raises_for_missing_column():
    data = pd.DataFrame({"x1": [1.0, None, 3.0]})
    handler = _make_handler(
        params={"strategies": {"unknown_var": MissingValueStrategy.MEAN.value}}
    )

    with pytest.raises(BadUserInput, match="were not found in the runtime data"):
        handler.transform_data(data=data)


@pytest.mark.parametrize(
    "strategy",
    [
        MissingValueStrategy.MEAN.value,
        MissingValueStrategy.MEDIAN.value,
        MissingValueStrategy.MOST_FREQUENT.value,
    ],
)
def test_transform_data_raises_insufficient_data_for_all_missing_column(strategy):
    data = pd.DataFrame({"x1": [None, None, None]})
    handler = _make_handler(params={"strategies": {"x1": strategy}})

    with pytest.raises(InsufficientDataError, match="has only missing values"):
        handler.transform_data(data=data)
