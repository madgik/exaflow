import pandas as pd
import pytest

from exaflow import exareme3_preprocessing_step_classes
from exaflow.algorithms.exareme3.preprocessing.outlier_winsorizer import (
    OutlierWinsorizer,
)
from exaflow.algorithms.federated.statistics.outlier_report import OutlierRule
from exaflow.algorithms.federated.statistics.outlier_report import OutlierStrategy
from exaflow.algorithms.federated.statistics.outlier_report import OutlierTail
from exaflow.algorithms.specifications import ParameterDictValueType
from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.worker_communication import BadUserInput
from exaflow.worker_communication import InsufficientDataError


def _make_winsorizer(params):
    return OutlierWinsorizer(params=params)


def _validate(winsorizer):
    winsorizer.validate_params(
        inputdata=Inputdata(
            data_model="dm:0.1",
            datasets=["d1"],
            x=["x1", "x2", "x_cat"],
            y=["y1"],
        ),
        metadata={
            "x1": {"is_categorical": False, "min": -100.0, "max": 100.0},
            "x2": {"is_categorical": False},
            "x_cat": {"is_categorical": True},
            "y1": {"is_categorical": False},
        },
    )


def test_preprocessing_step_is_discovered():
    assert (
        exareme3_preprocessing_step_classes["outlier_winsorizer"] is OutlierWinsorizer
    )


def test_get_specification_has_expected_shape():
    spec = OutlierWinsorizer.get_specification()

    assert spec.name == "outlier_winsorizer"
    assert spec.enabled is True
    assert set(spec.parameters.keys()) == {"strategies", "tails", "folds"}
    assert spec.parameters["strategies"].required is True
    assert spec.parameters["strategies"].dict_values_enums.source == [
        OutlierStrategy.GAUSSIAN.value,
        OutlierStrategy.IQR.value,
        OutlierStrategy.MAD.value,
        OutlierStrategy.QUANTILE.value,
    ]
    assert spec.parameters["tails"].dict_values_enums.source == [
        OutlierTail.LEFT.value,
        OutlierTail.RIGHT.value,
        OutlierTail.BOTH.value,
    ]
    assert spec.parameters["folds"].dict_values_type == ParameterDictValueType.REAL


def test_validate_params_accepts_per_variable_configuration():
    winsorizer = _make_winsorizer(
        {
            "strategies": {"x1": "iqr", "y1": "quantile"},
            "tails": {"x1": "left", "extra": "ignored"},
            "folds": {"y1": 0.1, "extra": "ignored"},
        }
    )

    _validate(winsorizer)


@pytest.mark.parametrize(
    "params, message",
    [
        ({"strategies": {"x_cat": "iqr"}}, "can only be used for numerical"),
        (
            {"strategies": {"x1": "iqr"}, "folds": {"x1": 0}},
            "Invalid outlier fold",
        ),
        (
            {"strategies": {"x1": "quantile"}, "folds": {"x1": 0.5}},
            "Invalid outlier fold",
        ),
    ],
)
def test_validate_params_rejects_invalid_configuration(params, message):
    winsorizer = _make_winsorizer(params)

    with pytest.raises(BadUserInput, match=message):
        _validate(winsorizer)


def test_transform_inputdata_variables_returns_identity():
    winsorizer = _make_winsorizer({"strategies": {"x1": "iqr"}})

    transformed_x, transformed_y = winsorizer.transform_inputdata_variables(
        x=["x1", "x2"],
        y=["y1"],
    )

    assert transformed_x == ["x1", "x2"]
    assert transformed_y == ["y1"]


def test_transform_metadata_returns_copy():
    metadata = {"x1": {"is_categorical": False}}
    winsorizer = _make_winsorizer({"strategies": {"x1": "iqr"}})

    transformed = winsorizer.transform_metadata(metadata=metadata)
    transformed["x1"]["is_categorical"] = True

    assert metadata["x1"]["is_categorical"] is False


def test_transform_metadata_promotes_configured_int_variable_to_real():
    metadata = {
        "x1": {"is_categorical": False, "sql_type": "int"},
        "x2": {"is_categorical": False, "sql_type": "int"},
    }
    winsorizer = _make_winsorizer({"strategies": {"x1": "iqr"}})

    transformed = winsorizer.transform_metadata(metadata=metadata)

    assert transformed["x1"]["sql_type"] == "real"
    assert transformed["x2"]["sql_type"] == "int"
    assert metadata["x1"]["sql_type"] == "int"


def test_clip_data_returns_same_dataframe_when_no_rules():
    data = pd.DataFrame({"v": [1.0, 2.0], "passthrough": ["a", "b"]})

    transformed, metadata_bounds = OutlierWinsorizer._clip_data(
        data=data,
        rules={},
        min_row_count=1,
    )

    assert transformed is data
    assert metadata_bounds == {}


def test_clip_data_mutates_initial_dataframe_and_passes_unconfigured_columns_through():
    data = pd.DataFrame(
        {
            "v": [-100.0, 10.0, 11.0, 12.0, 13.0, 14.0, 100.0],
            "passthrough": ["a", "b", "c", "d", "e", "f", "g"],
        }
    )

    transformed, metadata_bounds = OutlierWinsorizer._clip_data(
        data=data,
        rules={"v": OutlierRule("v", "iqr", "both", 1.5)},
        min_row_count=1,
    )

    assert transformed is data
    assert data["v"].tolist() == [6.0, 10.0, 11.0, 12.0, 13.0, 14.0, 18.0]
    assert data["passthrough"].tolist() == ["a", "b", "c", "d", "e", "f", "g"]
    assert metadata_bounds["v"] == (6.0, 18.0)


@pytest.mark.parametrize(
    "tail, expected_min, expected_max",
    [
        ("left", 6.0, 100.0),
        ("right", -100.0, 18.0),
        ("both", 6.0, 18.0),
    ],
)
def test_clip_data_respects_tail_selection(tail, expected_min, expected_max):
    data = pd.DataFrame(
        {
            "v": [-100.0, 10.0, 11.0, 12.0, 13.0, 14.0, 100.0],
        }
    )

    transformed, _ = OutlierWinsorizer._clip_data(
        data=data,
        rules={"v": OutlierRule("v", "iqr", tail, 1.5)},
        min_row_count=1,
    )

    assert transformed["v"].min() == expected_min
    assert transformed["v"].max() == expected_max


def test_clip_data_drops_remaining_nulls_in_configured_columns():
    data = pd.DataFrame(
        {
            "v": [None, 1, 2, 3, 4, 100],
            "other": ["dropped", "a", "b", "c", "d", "e"],
        }
    )

    transformed, _ = OutlierWinsorizer._clip_data(
        data=data,
        rules={"v": OutlierRule("v", "iqr", "both", 1.5)},
        min_row_count=1,
    )

    assert transformed is data
    assert transformed["v"].isna().sum() == 0
    assert transformed["other"].tolist() == ["a", "b", "c", "d", "e"]
    assert transformed["v"].tolist()[-1] < 100


def test_clip_data_raises_for_missing_configured_variable():
    data = pd.DataFrame({"v": [1, 2, 3]})

    with pytest.raises(BadUserInput, match="were not found in the runtime data"):
        OutlierWinsorizer._clip_data(
            data=data,
            rules={"missing": OutlierRule("missing", "iqr", "both", 1.5)},
            min_row_count=1,
        )


def test_clip_data_raises_insufficient_data_for_small_dataset():
    data = pd.DataFrame({"v": [1.0, 2.0]})

    with pytest.raises(InsufficientDataError, match="Insufficient non-missing data"):
        OutlierWinsorizer._clip_data(
            data=data,
            rules={"v": OutlierRule("v", "iqr", "both", 1.5)},
            min_row_count=3,
        )


def test_transform_data_and_metadata_updates_metadata_from_bounds(monkeypatch):
    from exaflow.worker import config as worker_config

    monkeypatch.setattr(worker_config.privacy, "minimum_row_count", 1)
    data = pd.DataFrame(
        {
            "x1": [10.0, 11.0, 12.0, 13.0, 14.0, 100.0],
        }
    )
    metadata = {
        "x1": {
            "is_categorical": False,
            "sql_type": "int",
            "min": -100.0,
            "max": 100.0,
        }
    }
    winsorizer = _make_winsorizer({"strategies": {"x1": "iqr"}})

    transformed, transformed_metadata = winsorizer.transform_data_and_metadata(
        data=data,
        metadata=metadata,
    )

    assert transformed["x1"].min() == 10.0
    assert transformed["x1"].max() == 17.5
    assert transformed_metadata["x1"]["min"] == 7.5
    assert transformed_metadata["x1"]["max"] == 17.5
    assert transformed_metadata["x1"]["sql_type"] == "real"
    assert metadata["x1"]["min"] == -100.0
