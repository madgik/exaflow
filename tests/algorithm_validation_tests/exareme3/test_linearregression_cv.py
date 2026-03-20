from pathlib import Path

import numpy as np
import pytest

from tests.algorithm_validation_tests.exareme3.helpers import get_test_params

expected_file = (
    Path(__file__).parent / "expected" / "linear_regression_cv_expected.json"
)
ALGNAME = "linear_regression_cv"


class TestLinearRegressionCV:
    @pytest.mark.parametrize("test_input, _", get_test_params(expected_file))
    def test_mean_abs_error(self, test_input, _, get_algorithm_result):
        result = get_algorithm_result(ALGNAME, test_input)
        mean_abs_error = np.array(result["mean_abs_error"])
        assert (0 <= mean_abs_error).all()

    @pytest.mark.parametrize("test_input, _", get_test_params(expected_file))
    def test_mean_sq_error(self, test_input, _, get_algorithm_result):
        result = get_algorithm_result(ALGNAME, test_input)
        mean_sq_error = np.array(result["mean_sq_error"])
        assert (0 <= mean_sq_error).all()

    @pytest.mark.parametrize("test_input, _", get_test_params(expected_file))
    def test_mean_r_squared(self, test_input, _, get_algorithm_result):
        result = get_algorithm_result(ALGNAME, test_input)
        r_squared = np.array(result["r_squared"])
        assert (0 <= r_squared).all()

    @pytest.mark.parametrize("test_input, _", get_test_params(expected_file))
    def test_mean_n_obs(self, test_input, _, get_algorithm_result):
        result = get_algorithm_result(ALGNAME, test_input)
        n_obs = np.array(result["n_obs"])
        assert (0 <= n_obs).all()
