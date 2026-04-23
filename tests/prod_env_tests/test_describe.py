from copy import deepcopy
from pathlib import Path

from tests.algorithm_validation_tests.exareme3.conftest import algorithm_request
from tests.algorithm_validation_tests.exareme3.conftest import parse_response
from tests.algorithm_validation_tests.exareme3.helpers import get_test_params


def _add_dropna_preprocessing(test_input):
    payload = deepcopy(test_input)
    inputdata = payload.get("inputdata", {})
    variables = [
        var
        for var in (inputdata.get("x", []) + inputdata.get("y", []))
        if var != "dataset"
    ]
    if not variables:
        return payload
    payload["preprocessing"] = {
        "missing_values_handler": {
            "strategies": {var: "drop" for var in variables},
        }
    }
    return payload


def _assert_describe_runs(expected_filename, add_dropna_preprocessing):
    expected_file = Path(__file__).parent / "expected" / expected_filename
    for test_input, expected in get_test_params(expected_file):
        payload = (
            _add_dropna_preprocessing(test_input)
            if add_dropna_preprocessing
            else test_input
        )
        response = algorithm_request("describe", payload, drop_na=False)
        result = parse_response(response)

        assert "featurewise" in result
        assert isinstance(result["featurewise"], list)
        assert len(result["featurewise"]) == len(expected)


def test_describe_featurewise():
    _assert_describe_runs(
        expected_filename="describe_featurewise_expected.json",
        add_dropna_preprocessing=False,
    )


def test_describe_analysis_set():
    _assert_describe_runs(
        expected_filename="describe_analysis_set_expected.json",
        add_dropna_preprocessing=True,
    )
