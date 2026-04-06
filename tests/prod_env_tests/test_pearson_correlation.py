from pathlib import Path

import pytest

from tests.algorithm_validation_tests.exareme3.conftest import algorithm_request
from tests.algorithm_validation_tests.exareme3.conftest import parse_response
from tests.algorithm_validation_tests.exareme3.helpers import get_test_params

algorithm_name = "pearson_correlation"

expected_file = Path(__file__).parent / "expected" / f"{algorithm_name}_expected.json"


@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_pearson_algorithm(test_input, expected):
    response = algorithm_request(algorithm_name, test_input)
    result = parse_response(response)

    # this test only ensures that the algorithm runs smoothly without errors
    assert result
