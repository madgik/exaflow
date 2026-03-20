from pathlib import Path

import numpy as np
import pytest

from tests.algorithm_validation_tests.exareme3.helpers import algorithm_request
from tests.algorithm_validation_tests.exareme3.helpers import get_test_params
from tests.algorithm_validation_tests.exareme3.helpers import parse_response

algorithm_name = "pearson_correlation"

expected_file = Path(__file__).parent / "expected" / f"{algorithm_name}_expected.json"


@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_pearson_algorithm(test_input, expected):
    response = algorithm_request(algorithm_name, test_input)
    result = parse_response(response)

    assert int(result["n_obs"]) == int(expected["n_obs"])
    for var in test_input["inputdata"]["y"]:
        np.testing.assert_allclose(
            result["correlations"][f"{var}"],
            expected["correlations"][f"{var}"],
            rtol=1e-7,
            atol=1e-10,
        )

        np.testing.assert_allclose(
            result["p_values"][f"{var}"],
            expected["p-values"][f"{var}"],
            rtol=1e-7,
            atol=1e-10,
        )

        np.testing.assert_allclose(
            result["ci_lo"][f"{var}"],
            expected["low_confidence_intervals"][f"{var}"],
            rtol=1e-7,
            atol=1e-10,
        )

        np.testing.assert_allclose(
            result["ci_hi"][f"{var}"],
            expected["high_confidence_intervals"][f"{var}"],
            rtol=1e-7,
            atol=1e-10,
        )


def assert_vectors_are_collinear(u, v):
    cosine_similarity = np.dot(v, u) / (np.sqrt(np.dot(v, v)) * np.sqrt(np.dot(u, u)))
    np.testing.assert_allclose(abs(cosine_similarity), 1, rtol=1e-7, atol=1e-10)
