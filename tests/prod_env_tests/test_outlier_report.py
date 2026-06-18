from pathlib import Path

import pytest

from tests.algorithm_validation_tests.exareme3.conftest import analysis_request
from tests.algorithm_validation_tests.exareme3.conftest import parse_response
from tests.algorithm_validation_tests.exareme3.helpers import get_test_params

algorithm_name = "outlier_report"
expected_file = Path(__file__).parent / "expected" / f"{algorithm_name}_expected.json"


RESPONSE_DATA_KEYS = {
    "strategy",
    "tail",
    "fold",
    "lower_bound",
    "upper_bound",
    "lower_outlier_count",
    "upper_outlier_count",
    "total_outlier_count",
    "total_outlier_percentage",
}

DESCRIBE_DATA_KEYS = {
    "mean",
    "std",
    "min",
    "max",
    "q1",
    "q2",
    "q3",
    "missing",
    "num_dtps",
    "total_count",
}


@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_outlier_report_validation(test_input, expected):
    response = analysis_request(
        algorithm_name,
        test_input,
        drop_na=expected.get("drop_na", True),
    )

    expected_status = expected.get("status_code", 200)
    assert response.status_code == expected_status, (
        f"{response.status_code}: {response.content}"
    )
    if expected_status != 200:
        assert expected["message_contains"].lower().encode() in response.content.lower()
        return

    result = parse_response(response)

    assert set(result.keys()) == {"featurewise"}
    assert result["featurewise"]
    assert {record["variable"] for record in result["featurewise"]} == set(
        expected["variables"]
    )
    assert {record["dataset"] for record in result["featurewise"]} == set(
        expected["datasets"]
    )
    assert len(result["featurewise"]) == (
        len(expected["variables"]) * len(expected["datasets"])
    )

    for record in result["featurewise"]:
        assert set(record.keys()) == {"variable", "dataset", "data"}
        assert set(record["data"].keys()) == RESPONSE_DATA_KEYS
        assert not (set(record["data"]) & DESCRIBE_DATA_KEYS)

    records_by_key = {
        (record["variable"], record["dataset"]): record
        for record in result["featurewise"]
    }
    for expected_record in expected["records"]:
        record = records_by_key[
            (expected_record["variable"], expected_record["dataset"])
        ]
        data = record["data"]
        assert data["strategy"] == expected_record["strategy"]
        assert data["tail"] == expected_record["tail"]
        assert data["fold"] == expected_record["fold"]

        if expected_record["tail"] == "left":
            assert data["upper_bound"] is None
            assert data["upper_outlier_count"] is None
            assert data["lower_bound"] is not None
        elif expected_record["tail"] == "right":
            assert data["lower_bound"] is None
            assert data["lower_outlier_count"] is None
            assert data["upper_bound"] is not None
        else:
            assert data["lower_bound"] is not None
            assert data["upper_bound"] is not None

        _assert_count_or_privacy_null(data["total_outlier_count"])
        _assert_percentage_or_privacy_null(data["total_outlier_percentage"])
        if data["lower_outlier_count"] is not None:
            assert isinstance(data["lower_outlier_count"], int)
        if data["upper_outlier_count"] is not None:
            assert isinstance(data["upper_outlier_count"], int)


def _assert_count_or_privacy_null(value):
    assert value is None or (isinstance(value, int) and value >= 0)


def _assert_percentage_or_privacy_null(value):
    assert value is None or (isinstance(value, float) and 0.0 <= value <= 100.0)
