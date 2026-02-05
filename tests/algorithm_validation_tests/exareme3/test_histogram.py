from pathlib import Path

import pytest

from tests.algorithm_validation_tests.exareme3.helpers import algorithm_request
from tests.algorithm_validation_tests.exareme3.helpers import get_test_params
from tests.algorithm_validation_tests.exareme3.helpers import parse_response

algorithm_name = "histogram"

expected_file = Path(__file__).parent / "expected" / f"{algorithm_name}_expected.json"


@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_histogram(test_input, expected):
    response = algorithm_request(algorithm_name, test_input)
    result = parse_response(response)

    # sort records by variable and dataset in order to compare them

    assert expected["histogram"] == result["histogram"]
