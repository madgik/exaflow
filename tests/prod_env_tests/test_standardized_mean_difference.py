import functools
from pathlib import Path

import numpy as np
import pytest

from tests.algorithm_validation_tests.exareme3.conftest import analysis_request
from tests.algorithm_validation_tests.exareme3.conftest import parse_response
from tests.algorithm_validation_tests.exareme3.helpers import get_test_params

algorithm_name = "standardized_mean_difference"
expected_file = Path(__file__).parent / "expected" / f"{algorithm_name}_expected.json"
assert_allclose = functools.partial(np.testing.assert_allclose, rtol=1e-7, atol=1e-10)


@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_standardized_mean_difference(test_input, expected):
    response = analysis_request(algorithm_name, test_input)
    result = parse_response(response)

    actual = result["comparisons"]
    expected_comparisons = expected["comparisons"]
    assert len(actual) == len(expected_comparisons)
    for actual_pair, expected_pair in zip(actual, expected_comparisons):
        assert set(actual_pair) == {"group1", "group2", "smd"}
        assert actual_pair["group1"] == expected_pair["group1"]
        assert actual_pair["group2"] == expected_pair["group2"]
        assert_allclose(actual_pair["smd"], expected_pair["smd"])
