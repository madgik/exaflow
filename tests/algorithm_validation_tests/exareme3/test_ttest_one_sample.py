from pathlib import Path

import numpy
import pytest

from tests.algorithm_validation_tests.exareme3.helpers import analysis_request
from tests.algorithm_validation_tests.exareme3.helpers import assert_allclose
from tests.algorithm_validation_tests.exareme3.helpers import get_test_params
from tests.algorithm_validation_tests.exareme3.helpers import parse_response

algorithm_name = "ttest_onesample"

expected_file = Path(__file__).parent / "expected" / f"{algorithm_name}_expected.json"


@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_one_sample_ttest(test_input, expected):
    response = analysis_request(algorithm_name, test_input)
    result = parse_response(response)
    mu = test_input["algorithm"]["parameters"]["mu"]

    assert_allclose(result["n_obs"], expected["n_obs"], rtol=1e-8, atol=1e-10)
    assert_allclose(result["t_stat"], expected["t_value"], rtol=1e-8, atol=1e-10)
    assert_allclose(result["p"], expected["p_value"], rtol=1e-8, atol=1e-10)
    assert_allclose(result["df"], expected["df"], rtol=1e-8, atol=1e-10)
    assert_allclose(
        result["mean_diff"], expected["mean_diff"] - mu, rtol=1e-8, atol=1e-10
    )
    assert_allclose(result["se_diff"], expected["se_diff"], rtol=1e-8, atol=1e-10)
    assert_allclose(
        numpy.abs(result["cohens_d"]),
        numpy.abs(expected["cohens_d"]),
        rtol=1e-8,
        atol=1e-10,
    )
    # R reports the confidence interval for the mean; Exaflow reports it for
    # mean - mu.
    assert_allclose(
        float(result["ci_upper"]), expected["ci_upper"] - mu, rtol=1e-8, atol=1e-10
    )
    assert_allclose(
        float(result["ci_lower"]), expected["ci_lower"] - mu, rtol=1e-8, atol=1e-10
    )
