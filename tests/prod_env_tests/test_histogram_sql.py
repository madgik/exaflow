import json
import re
from pathlib import Path

import pytest
import requests

from tests.algorithm_validation_tests.exareme3.conftest import algorithm_request
from tests.algorithm_validation_tests.exareme3.conftest import parse_response
from tests.algorithm_validation_tests.exareme3.helpers import get_test_params
from tests.prod_env_tests import algorithms_url

algorithm_name = "histogram_sql"

expected_file = Path(__file__).parent / "expected" / f"{algorithm_name}_expected.json"


@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_histogram_sql(test_input, expected):
    # Use SimpleHistogram for production environment tests
    if "parameters" not in test_input:
        test_input["parameters"] = {}
    test_input["parameters"]["histogram_type"] = "simple"

    response = algorithm_request(algorithm_name, test_input, drop_na=False)
    result = parse_response(response)

    # this test only ensures that the algorithm runs smoothly without errors
    assert result


def test_histogram_sql_insufficient_data_after_filters_returns_461():
    request_dict = {
        "inputdata": {
            "data_model": "dementia:0.1",
            "datasets": ["edsd0"],
            "y": ["lefthippocampus"],
            "filters": {
                "condition": "AND",
                "rules": [
                    {
                        "condition": "OR",
                        "rules": [
                            {
                                "id": "subjectage",
                                "field": "subjectage",
                                "type": "real",
                                "input": "number",
                                "operator": "greater",
                                "value": 200.0,
                            }
                        ],
                    }
                ],
                "valid": True,
            },
        },
        "preprocessing": {
            "missing_values_handler": {"strategies": {"lefthippocampus": "drop"}}
        },
        "parameters": {"bins": 20, "histogram_type": "simple"},
        "type": "exareme3",
    }
    algorithm_url = algorithms_url + "/histogram_sql"
    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.post(
        algorithm_url, data=json.dumps(request_dict), headers=headers
    )

    assert response.status_code == 461, f"Response message: {response.text}"
    assert re.search(
        "The algorithm could not run with the input provided because there are insufficient data.",
        response.text,
    )
