from pathlib import Path

import pytest

from tests.algorithm_validation_tests.exareme3.conftest import algorithm_request
from tests.algorithm_validation_tests.exareme3.conftest import parse_response
from tests.algorithm_validation_tests.exareme3.helpers import get_test_params

algorithm_name = "ttest_welch"
expected_file = Path(__file__).parent / "expected" / f"{algorithm_name}_expected.json"


@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_ttest_welch_validation(test_input, expected):
    response = algorithm_request(algorithm_name, test_input)
    result = parse_response(response)

    expected_keys = expected.get(
        "required_keys",
        [
            "t_stat",
            "df",
            "p",
            "mean_diff",
            "se_diff",
            "ci_upper",
            "ci_lower",
            "cohens_d",
        ],
    )
    for key in expected_keys:
        assert key in result

    assert float(result["df"]) > 0.0
    assert 0.0 <= float(result["p"]) <= 1.0
    assert float(result["se_diff"]) > 0.0
