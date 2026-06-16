from pathlib import Path

import pytest

from tests.algorithm_validation_tests.exareme3.conftest import algorithm_request
from tests.algorithm_validation_tests.exareme3.conftest import parse_response
from tests.algorithm_validation_tests.exareme3.helpers import get_test_params

algorithm_name = "linear_regression"

expected_file = Path(__file__).parent / "expected" / f"{algorithm_name}_expected.json"


@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_linearregression_algorithm(test_input, expected):
    response = algorithm_request(algorithm_name, test_input)
    result = parse_response(response)

    # this test only ensures that the algorithm runs smoothly without errors
    assert result


def test_linearregression_perfect_fit_serializes_infinite_f_stat_as_null():
    test_input = {
        "inputdata": {
            "variables": ["rightocpoccipitalpole"],
            "data_model": "dementia:0.1",
            "datasets": [
                "edsd7",
                "edsd4",
                "ppmi2",
                "desd-synthdata2",
                "edsd0",
                "ppmi5",
                "desd-synthdata8",
                "ppmi3",
            ],
            "filters": None,
        },
        "algorithm": {
            "x": ["rightocpoccipitalpole"],
            "y": ["rightocpoccipitalpole"],
            "parameters": {},
        },
    }

    response = algorithm_request(algorithm_name, test_input)
    assert response.status_code == 460, response.text
    assert (
        "Inputdata 'x' and 'y' should not contain the same variables." in response.text
    )
