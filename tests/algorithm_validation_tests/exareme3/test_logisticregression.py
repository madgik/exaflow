from pathlib import Path

import numpy as np
import pytest

from tests.algorithm_validation_tests.exareme3.helpers import assert_allclose
from tests.algorithm_validation_tests.exareme3.helpers import get_test_params

ALGNAME = "logistic_regression"

expected_file = Path(__file__).parent / "expected" / "logistic_regression_expected.json"


class TestLogisticRegression:
    @pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
    def test_n_obs(self, test_input, expected, get_algorithm_result):
        result = get_algorithm_result(ALGNAME, test_input)["summary"]
        assert result["n_obs"] == expected["n_obs"]

    @pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
    def test_df_model(self, test_input, expected, get_algorithm_result):
        result = get_algorithm_result(ALGNAME, test_input)["summary"]
        assert result["df_model"] == expected["df_model"]

    @pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
    def test_df_resid(self, test_input, expected, get_algorithm_result):
        result = get_algorithm_result(ALGNAME, test_input)["summary"]
        assert result["df_resid"] == expected["df_resid"]

    @pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
    def test_aic(self, test_input, expected, get_algorithm_result):
        result = get_algorithm_result(ALGNAME, test_input)["summary"]
        assert_allclose(result["aic"], expected["aic"])

    @pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
    def test_bic(self, test_input, expected, get_algorithm_result):
        result = get_algorithm_result(ALGNAME, test_input)["summary"]
        assert_allclose(result["bic"], expected["bic"])

    @pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
    def test_loglikelihood(self, test_input, expected, get_algorithm_result):
        result = get_algorithm_result(ALGNAME, test_input)["summary"]
        assert_allclose(result["ll"], expected["ll"])

    @pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
    def test_loglikelihood0(self, test_input, expected, get_algorithm_result):
        result = get_algorithm_result(ALGNAME, test_input)["summary"]
        assert_allclose(result["ll0"], expected["ll0"])

    @pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
    def test_r_squared_mcf(self, test_input, expected, get_algorithm_result):
        result = get_algorithm_result(ALGNAME, test_input)["summary"]
        assert_allclose(result["r_squared_mcf"], expected["r_squared_mcf"])

    @pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
    def test_coefficients(self, test_input, expected, get_algorithm_result):
        result = get_algorithm_result(ALGNAME, test_input)["summary"]
        assert_allclose(result["coefficients"], expected["coefficients"])

    # some quantities need a higher tolerance due to error propagation effects
    @pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
    def test_stderr(self, test_input, expected, get_algorithm_result):
        result = get_algorithm_result(ALGNAME, test_input)["summary"]
        np.testing.assert_allclose(
            result["stderr"], expected["stderr"], rtol=1e-5, atol=1e-5
        )

    @pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
    def test_z_scores(self, test_input, expected, get_algorithm_result):
        result = get_algorithm_result(ALGNAME, test_input)["summary"]
        np.testing.assert_allclose(
            result["z_scores"], expected["z_scores"], rtol=1e-5, atol=1e-5
        )

    @pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
    def test_pvalues(self, test_input, expected, get_algorithm_result):
        result = get_algorithm_result(ALGNAME, test_input)["summary"]
        np.testing.assert_allclose(
            result["pvalues"], expected["pvalues"], rtol=1e-5, atol=1e-5
        )

    @pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
    def test_lower_ci(self, test_input, expected, get_algorithm_result):
        result = get_algorithm_result(ALGNAME, test_input)["summary"]
        np.testing.assert_allclose(
            result["lower_ci"], expected["lower_ci"], rtol=1e-5, atol=1e-5
        )

    @pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
    def test_upper_ci(self, test_input, expected, get_algorithm_result):
        result = get_algorithm_result(ALGNAME, test_input)["summary"]
        np.testing.assert_allclose(
            result["upper_ci"], expected["upper_ci"], rtol=1e-5, atol=1e-5
        )
