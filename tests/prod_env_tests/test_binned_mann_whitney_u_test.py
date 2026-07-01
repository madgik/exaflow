import functools
from pathlib import Path

import numpy as np
import pytest

from tests.algorithm_validation_tests.exareme3.conftest import analysis_request
from tests.algorithm_validation_tests.exareme3.conftest import parse_response
from tests.algorithm_validation_tests.exareme3.helpers import get_test_params

algorithm_name = "binned_mann_whitney_u_test"
expected_file = Path(__file__).parent / "expected" / f"{algorithm_name}_expected.json"

assert_allclose_approx = functools.partial(
    np.testing.assert_allclose,
    rtol=2e-3,
    atol=1e-6,
)


@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_binned_mann_whitney_u_test(test_input, expected):
    response = analysis_request(algorithm_name, test_input)
    result = parse_response(response)

    assert result["n1"] == expected["n1"]
    assert result["n2"] == expected["n2"]
    assert_allclose_approx(result["u_stat"], expected["u_stat"])
    assert_allclose_approx(result["p_value"], expected["p_value"])
    assert_allclose_approx(result["z_score"], expected["z_score"])
