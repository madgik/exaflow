from pathlib import Path

import pytest

from tests.algorithm_validation_tests.exareme3.conftest import analysis_request
from tests.algorithm_validation_tests.exareme3.conftest import parse_response
from tests.algorithm_validation_tests.exareme3.helpers import assert_allclose
from tests.algorithm_validation_tests.exareme3.helpers import get_test_params

algorithm_name = "quartiles"
expected_file = Path(__file__).parent / "expected" / f"{algorithm_name}_expected.json"


@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_quartiles_validation(test_input, expected):
    response = analysis_request(algorithm_name, test_input)
    result = parse_response(response)

    expected_quantiles = expected["quantiles"]
    if expected_quantiles is None:
        assert result["quantiles"] is None
        return

    quantiles = result["quantiles"]
    assert quantiles is not None
    assert len(quantiles) == len(expected_quantiles) == 3
    for actual, exp in zip(quantiles, expected_quantiles):
        assert actual["q"] == exp["q"]
        if exp["value"] is None:
            assert actual["value"] is None
            assert actual["actual_q"] is None
        else:
            assert isinstance(actual["value"], (int, float))
            assert isinstance(actual["actual_q"], (int, float))
            assert_allclose(actual["value"], exp["value"])
            assert_allclose(actual["actual_q"], exp["actual_q"])
