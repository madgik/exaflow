import pandas as pd
import pytest

from exaflow.algorithms.exareme3.missing_values_handler import STRATEGY_DROP
from exaflow.algorithms.exareme3.missing_values_handler import MissingValuesHandler
from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.worker_communication import BadUserInput


def _make_handler(strategy=STRATEGY_DROP):
    return MissingValuesHandler(params={"strategy": strategy})


def test_get_specification_has_expected_shape():
    spec = MissingValuesHandler.get_specification()

    assert spec.name == "missing_values_handler"
    assert spec.enabled is True
    assert set(spec.parameters.keys()) == {"strategy"}
    assert spec.parameters["strategy"].enums.source == [STRATEGY_DROP]


def test_validate_params_accepts_drop_strategy():
    handler = _make_handler()

    handler.validate_params(
        inputdata=Inputdata(
            data_model="dm:0.1",
            datasets=["d1"],
            x=["x1"],
            y=["y1"],
        ),
        metadata={"x1": {"is_categorical": False}, "y1": {"is_categorical": False}},
    )


def test_validate_params_rejects_unknown_strategy():
    handler = _make_handler(strategy="unknown")

    with pytest.raises(BadUserInput, match="Invalid strategy"):
        handler.validate_params(
            inputdata=Inputdata(
                data_model="dm:0.1",
                datasets=["d1"],
                x=["x1"],
                y=["y1"],
            ),
            metadata={"x1": {"is_categorical": False}, "y1": {"is_categorical": False}},
        )


def test_transform_inputdata_variables_returns_identity():
    handler = _make_handler()

    transformed_x, transformed_y = handler.transform_inputdata_variables(
        x=["x1", "x2"],
        y=["y1"],
    )

    assert transformed_x == ["x1", "x2"]
    assert transformed_y == ["y1"]


def test_transform_metadata_returns_copy():
    metadata = {"x1": {"is_categorical": False}}
    handler = _make_handler()

    transformed = handler.transform_metadata(metadata=metadata)
    transformed["x1"]["is_categorical"] = True

    assert metadata["x1"]["is_categorical"] is False


def test_transform_data_drop_strategy_removes_rows_with_any_missing_values():
    data = pd.DataFrame(
        {
            "x1": [1.0, None, 3.0],
            "x2": [10.0, 20.0, None],
            "dataset": ["d1", "d1", "d1"],
        }
    )
    handler = _make_handler()

    transformed = handler.transform_data(data=data)

    expected = pd.DataFrame({"x1": [1.0], "x2": [10.0], "dataset": ["d1"]})
    pd.testing.assert_frame_equal(
        transformed.reset_index(drop=True),
        expected,
        check_dtype=False,
    )
