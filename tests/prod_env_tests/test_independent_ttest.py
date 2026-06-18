from pathlib import Path

import pytest

from tests.algorithm_validation_tests.exareme3.conftest import analysis_request
from tests.algorithm_validation_tests.exareme3.conftest import parse_response
from tests.algorithm_validation_tests.exareme3.helpers import get_test_params

algorithm_name = "ttest_independent"

expected_file = Path(__file__).parent / "expected" / f"{algorithm_name}_expected.json"


@pytest.mark.parametrize(
    "test_input, expected",
    get_test_params(
        expected_file,
        skip_indices=[4],
        skip_reason="P-value assertion fail. Related ticket: https://team-1617704806227.atlassian.net/browse/MIP-794",
    ),
)
def test_independent_ttest(test_input, expected, subtests):
    response = analysis_request(algorithm_name, test_input)
    result = parse_response(response)

    # this test only ensures that the algorithm runs smoothly without errors
    assert result
