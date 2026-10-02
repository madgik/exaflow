from pathlib import Path

import numpy as np
import pytest

from tests.algorithm_validation_tests.exareme3.conftest import analysis_request
from tests.algorithm_validation_tests.exareme3.conftest import parse_response
from tests.algorithm_validation_tests.exareme3.helpers import get_test_params

expected_file = (
    Path(__file__).parent / "expected" / "linear_regression_cv_expected.json"
)


@pytest.mark.parametrize("test_input, _", get_test_params(expected_file))
def test_linear_regression_cv(test_input, _):
    response = analysis_request("linear_regression_cv", test_input)
    result = parse_response(response)
    assert "mean_sq_error" not in result
    rmse = result["root_mean_sq_error"]
    mae = result["mean_abs_error"]
    assert np.isfinite(rmse + mae).all()
    # These fixtures use distinct, non-perfect outcome/predictor pairs.
    assert rmse[0] >= mae[0] > 0
    assert rmse[1] >= 0
    assert mae[1] >= 0
    assert np.isfinite(result["r_squared"]).all()
    assert result["r_squared"][0] <= 1
    assert result["r_squared"][1] >= 0
    assert len(result["n_obs"]) == test_input["algorithm"]["parameters"]["n_splits"]
    assert all(n > 0 for n in result["n_obs"])
