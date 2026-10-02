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
        assert np.isfinite(mean_abs_error).all()
        assert mean_abs_error[0] > 0
        assert mean_abs_error[1] >= 0

    @pytest.mark.parametrize("test_input, _", get_test_params(expected_file))
    def test_root_mean_sq_error(self, test_input, _, get_algorithm_result):
        result = get_algorithm_result(ALGNAME, test_input)
        assert "mean_sq_error" not in result
        rmse = np.array(result["root_mean_sq_error"])
        assert np.isfinite(rmse).all()
        assert rmse[0] > 0
        assert rmse[1] >= 0
        assert rmse[0] >= result["mean_abs_error"][0]

    @pytest.mark.parametrize("test_input, _", get_test_params(expected_file))
    def test_mean_r_squared(self, test_input, _, get_algorithm_result):
        result = get_algorithm_result(ALGNAME, test_input)
        r_squared = np.array(result["r_squared"])
        assert np.isfinite(r_squared).all()
        assert r_squared[0] <= 1
        assert r_squared[1] >= 0

    @pytest.mark.parametrize("test_input, _", get_test_params(expected_file))
    def test_mean_n_obs(self, test_input, _, get_algorithm_result):
        result = get_algorithm_result(ALGNAME, test_input)
        n_obs = np.array(result["n_obs"])
        assert len(n_obs) == test_input["algorithm"]["parameters"]["n_splits"]
        assert (0 < n_obs).all()
