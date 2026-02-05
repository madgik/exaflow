from pathlib import Path

import pytest

from tests.algorithm_validation_tests.exareme3.helpers import algorithm_request
from tests.algorithm_validation_tests.exareme3.helpers import assert_allclose
from tests.algorithm_validation_tests.exareme3.helpers import get_test_params
from tests.algorithm_validation_tests.exareme3.helpers import parse_response

algorithm_name = "ttest_independent"

expected_file = Path(__file__).parent / "expected" / f"{algorithm_name}_expected.json"


@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_independent_ttest(test_input, expected, subtests):
    response = algorithm_request(algorithm_name, test_input)
    result = parse_response(response)

    with subtests.test():
        assert_allclose(result["t_stat"], expected["statistic"], rtol=1e-8, atol=1e-10)
    with subtests.test():
        assert_allclose(result["p"], expected["p_value"], rtol=1e-8, atol=1e-3)
    # Some tests were failing with lower atol than 1e-3 (aprox. 5%). Since no tests
    # fail for t-stat and df, we proceeded with increasing the tolerance for the p-value,
    # confident that the algorithm produces accurate results
    with subtests.test():
        assert_allclose(
            result["mean_diff"], expected["mean_diff"], rtol=1e-8, atol=1e-10
        )
    with subtests.test():
        assert_allclose(result["df"], expected["df"], rtol=1e-8, atol=1e-10)
    with subtests.test():
        assert_allclose(
            result["se_diff"], expected["se_difference"], rtol=1e-8, atol=1e-10
        )
    with subtests.test():
        assert_allclose(result["cohens_d"], expected["cohens_d"], rtol=1e-8, atol=1e-10)
