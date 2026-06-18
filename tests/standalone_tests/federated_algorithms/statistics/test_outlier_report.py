import math

import pandas as pd
import pytest

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.statistics.outlier_report import OutlierReport
from exaflow.algorithms.federated.statistics.outlier_report import (
    FederatedOutlierReport,
)
from exaflow.algorithms.federated.statistics.outlier_report import OutlierRule
from exaflow.algorithms.federated.statistics.outlier_report import WinsorizationHelper
from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.worker_communication import BadUserInput
from exaflow.worker_communication import InsufficientDataError


def _make_algorithm(params):
    return OutlierReport(
        engine=None,
        inputdata=Inputdata(
            data_model="dm:0.1",
            datasets=["d1"],
            variables=["x1", "x_cat", "y1"],
        ),
        x=["x1", "x_cat"],
        y=["y1"],
        metadata={
            "x1": {"is_categorical": False},
            "x_cat": {"is_categorical": True},
            "y1": {"is_categorical": False},
        },
        parameters=params,
    )


def _validate(algorithm):
    algorithm._validate_parameters(
        strategies=algorithm._dict_parameter("strategies"),
        folds=algorithm._dict_parameter("folds"),
    )


def test_get_specification_has_expected_shape():
    spec = OutlierReport.get_specification()

    assert spec.name == "outlier_report"
    assert spec.enabled is True
    assert set(spec.parameters.keys()) == {"strategies", "tails", "folds"}
    assert spec.parameters["strategies"].required is True
    assert spec.parameters["strategies"].dict_values_enums.source == [
        "gaussian",
        "iqr",
        "mad",
        "quantile",
    ]
    assert spec.parameters["tails"].dict_values_enums.source == [
        "left",
        "right",
        "both",
    ]
    assert (
        spec.parameters["folds"].dict_values_type == specs.ParameterDictValueType.REAL
    )


def test_validate_params_accepts_per_variable_configuration():
    algorithm = _make_algorithm(
        {
            "strategies": {"x1": "iqr", "y1": "quantile"},
            "tails": {"x1": "left", "extra": "ignored"},
            "folds": {"y1": 0.1, "extra": "ignored"},
        }
    )

    _validate(algorithm)


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
        (
            {"strategies": {"x1": "iqr"}, "folds": {"x1": "nan"}},
            "Invalid outlier fold",
        ),
        (
            {"strategies": {"x1": "gaussian"}, "folds": {"x1": "inf"}},
            "Invalid outlier fold",
        ),
        (
            {"strategies": {"x1": "mad"}, "folds": {"x1": "-inf"}},
            "Invalid outlier fold",
        ),
    ],
)
def test_validate_params_rejects_invalid_configuration(params, message):
    algorithm = _make_algorithm(params)

    with pytest.raises(BadUserInput, match=message):
        _validate(algorithm)


def test_get_specification_accepts_only_numerical_variables():
    spec = OutlierReport.get_specification()

    assert spec.y.types == [
        specs.InputDataType.REAL,
        specs.InputDataType.INT,
    ]
    assert spec.x.types == [
        specs.InputDataType.REAL,
        specs.InputDataType.INT,
    ]
    assert spec.y.stattypes == [specs.InputDataStatType.NUMERICAL]
    assert spec.x.stattypes == [specs.InputDataStatType.NUMERICAL]


@pytest.mark.parametrize(
    "rule, expected_lower, expected_upper",
    [
        (
            OutlierRule("v", "gaussian", "both", 2.0),
            -1.1622776601683795,
            5.16227766016838,
        ),
        (OutlierRule("v", "iqr", "both", 1.5), -2.0, 6.0),
        (OutlierRule("v", "mad", "both", 2.0), -0.9652, 4.9652),
        (OutlierRule("v", "quantile", "both", 0.2), 0.8, 3.2),
    ],
)
def test_compute_bounds_for_supported_strategies(rule, expected_lower, expected_upper):
    series = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0])

    bounds = WinsorizationHelper.compute_bounds(series, rule)

    assert math.isclose(bounds.lower, expected_lower, abs_tol=1e-6)
    assert math.isclose(bounds.upper, expected_upper, abs_tol=1e-6)


@pytest.mark.parametrize(
    "tail, lower_count, upper_count, total_count",
    [
        ("left", 2, None, 2),
        ("right", None, 2, 2),
        ("both", 2, 2, 4),
    ],
)
def test_report_respects_tail_selection(tail, lower_count, upper_count, total_count):
    data = pd.DataFrame(
        {
            "dataset": ["d1"] * 7,
            "v": [-100.0, 10.0, 11.0, 12.0, 13.0, 14.0, 100.0],
        }
    )
    report = FederatedOutlierReport()

    records = report.report(
        data=data,
        rules={"v": OutlierRule("v", "quantile", tail, 0.2)},
        min_row_count=1,
    )

    payload = records[0]["data"]
    assert payload["lower_outlier_count"] == lower_count
    assert payload["upper_outlier_count"] == upper_count
    assert payload["total_outlier_count"] == total_count


def test_report_masks_small_non_zero_counts_and_total():
    data = pd.DataFrame(
        {
            "dataset": ["d1"] * 7,
            "v": [-100.0, 10.0, 11.0, 12.0, 13.0, 14.0, 100.0],
        }
    )
    report = FederatedOutlierReport()

    records = report.report(
        data=data,
        rules={"v": OutlierRule("v", "quantile", "both", 0.2)},
        min_row_count=3,
    )

    payload = records[0]["data"]
    assert payload["lower_outlier_count"] is None
    assert payload["upper_outlier_count"] is None
    assert payload["total_outlier_count"] is None
    assert payload["total_outlier_percentage"] is None


def test_report_keeps_zero_counts_visible():
    data = pd.DataFrame({"dataset": ["d1"] * 5, "v": [10, 11, 12, 13, 14]})
    report = FederatedOutlierReport()

    records = report.report(
        data=data,
        rules={"v": OutlierRule("v", "iqr", "both", 1.5)},
        min_row_count=3,
    )

    payload = records[0]["data"]
    assert payload["lower_outlier_count"] == 0
    assert payload["upper_outlier_count"] == 0
    assert payload["total_outlier_count"] == 0
    assert payload["total_outlier_percentage"] == 0.0


def test_report_raises_insufficient_data_for_small_dataset():
    data = pd.DataFrame({"dataset": ["d1", "d1"], "v": [1.0, 2.0]})
    report = FederatedOutlierReport()

    with pytest.raises(InsufficientDataError, match="Insufficient non-missing data"):
        report.report(
            data=data,
            rules={"v": OutlierRule("v", "iqr", "both", 1.5)},
            min_row_count=3,
        )


def test_report_returns_local_dataset_records_without_describe_fields():
    data = pd.DataFrame(
        {
            "dataset": ["d1", "d1", "d1", "d2", "d2", "d2"],
            "v": [1.0, 2.0, 100.0, 10.0, 11.0, 12.0],
        }
    )
    report = FederatedOutlierReport()

    records = report.report(
        data=data,
        rules={"v": OutlierRule("v", "iqr", "both", 1.5)},
        min_row_count=1,
    )

    assert [record["dataset"] for record in records] == ["d1", "d2"]
    payload_keys = set(records[0]["data"])
    assert {
        "strategy",
        "tail",
        "fold",
        "lower_bound",
        "upper_bound",
        "lower_outlier_count",
        "upper_outlier_count",
        "total_outlier_count",
        "total_outlier_percentage",
    } == payload_keys
    assert (
        not {"mean", "std", "min", "max", "q1", "q2", "q3", "num_dtps"} & payload_keys
    )
