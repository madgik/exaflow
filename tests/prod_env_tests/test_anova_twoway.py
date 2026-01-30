from pathlib import Path

import pytest

from tests.algorithm_validation_tests.exareme3.conftest import algorithm_request
from tests.algorithm_validation_tests.exareme3.conftest import parse_response
from tests.algorithm_validation_tests.exareme3.helpers import get_test_params

expected_file = Path(__file__).parent / "expected" / "anova_twoway_expected.json"


@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_anova_two_way(test_input, expected, subtests):
    response = algorithm_request("anova_twoway", test_input)
    result = parse_response(response)

    # this test only ensures that the algorithm runs smoothly without errors
    assert result
