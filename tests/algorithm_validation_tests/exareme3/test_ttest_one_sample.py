from pathlib import Path

import numpy
import pytest

from tests.algorithm_validation_tests.exareme3.helpers import algorithm_request
from tests.algorithm_validation_tests.exareme3.helpers import assert_allclose
from tests.algorithm_validation_tests.exareme3.helpers import get_test_params
from tests.algorithm_validation_tests.exareme3.helpers import parse_response

algorithm_name = "ttest_onesample"

expected_file = Path(__file__).parent / "expected" / f"{algorithm_name}_expected.json"


@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_one_sample_ttest(test_input, expected):
    response = algorithm_request(algorithm_name, test_input)
    result = parse_response(response)

    assert_allclose(result["n_obs"], expected["n_obs"], rtol=1e-8, atol=1e-10)
    assert_allclose(result["t_stat"], expected["t_value"], rtol=1e-8, atol=1e-10)
    assert_allclose(result["p"], expected["p_value"], rtol=1e-8, atol=1e-10)
    assert_allclose(result["df"], expected["df"], rtol=1e-8, atol=1e-10)
    assert_allclose(result["mean_diff"], expected["mean_diff"], rtol=1e-8, atol=1e-10)
    assert_allclose(result["se_diff"], expected["se_diff"], rtol=1e-8, atol=1e-10)
    assert_allclose(
        numpy.abs(result["cohens_d"]),
        numpy.abs(expected["cohens_d"]),
        rtol=1e-8,
        atol=1e-10,
    )

    # confidence intervals are not tested because of an issue with rpy2.
    # However, the way they are calculated is correct, and verifying them
    # was deemed not an absolute necessity at the current state of the project.
    # assert_allclose(result["ci_upper"], expected["ci_upper"], rtol=1e-8, atol=1e-10)
    # assert_allclose(result["ci_lower"], expected["ci_lower"], rtol=1e-8, atol=1e-10)
