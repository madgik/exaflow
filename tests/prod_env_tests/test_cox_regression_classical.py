import json
import re
from pathlib import Path

import pytest

from tests.algorithm_validation_tests.exareme3.conftest import analysis_request
from tests.algorithm_validation_tests.exareme3.conftest import parse_response
from tests.algorithm_validation_tests.exareme3.helpers import get_test_params

algorithm_name = "cox_regression_classical"

expected_file = Path(__file__).parent / "expected" / f"{algorithm_name}_expected.json"


def _load_test_cases():
    with expected_file.open() as f:
        return json.load(f)["test_cases"]


def _longitudinal_transformer_parameters(request: dict) -> dict:
    for step in request["preprocessing"]:
        if step["name"] == "longitudinal_transformer":
            return step["parameters"]
    raise AssertionError("longitudinal_transformer preprocessing step missing")


@pytest.mark.parametrize("test_input, _expected", get_test_params(expected_file))
def test_cox_regression_classical_algorithm(test_input, _expected):
    response = analysis_request(algorithm_name, test_input)
    result = parse_response(response)
    summary = result.get("summary", {})
    print(
        "[prod cox_regression_classical]",
        f"n_obs={summary.get('n_obs')}",
        f"n_events={summary.get('n_events')}",
        f"event_times={summary.get('n_unique_event_times')}",
        f"converged={summary.get('converged')}",
        f"method={summary.get('method')}",
    )
    assert result


def test_cox_regression_classical_expected_fixture_has_10_real_cases():
    test_cases = _load_test_cases()
    assert len(test_cases) == 10
    for case in test_cases:
        request = case["input"]
        assert request["inputdata"]["data_model"] == "longitudinal_dementia:0.1"
        assert request["algorithm"]["y"] == ["timesincebaseline"]
        assert (
            request["algorithm"]["parameters"]["event_var"] == "alzheimerbroadcategory"
        )
        transformer = _longitudinal_transformer_parameters(request)
        assert transformer["visit1"] == "BL"
        assert transformer["visit2"] == "FL2"


def test_cox_regression_classical_invalid_positive_class():
    request = _load_test_cases()[0]["input"]
    request = json.loads(json.dumps(request))
    request["algorithm"]["parameters"]["positive_class"] = "NOT_A_REAL_EVENT_LEVEL"

    response = analysis_request(algorithm_name, request)
    assert response.status_code == 460, response.text
    assert re.search(
        r"positive_class.*observed event variable levels",
        response.text,
    ), response.text


def test_cox_regression_classical_rejects_multiple_time_variables():
    request = _load_test_cases()[0]["input"]
    request = json.loads(json.dumps(request))
    request["inputdata"]["variables"].append("righthippocampus")
    request["algorithm"]["y"] = ["timesincebaseline", "righthippocampus"]
    transformer = _longitudinal_transformer_parameters(request)
    transformer["strategies"]["righthippocampus"] = "first"

    response = analysis_request(algorithm_name, request)
    assert response.status_code == 460, response.text
    assert re.search(r"Follow-up time.*at most 1 values", response.text), response.text
