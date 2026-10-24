from pathlib import Path

import pytest

from tests.algorithm_validation_tests.exareme3.conftest import analysis_request
from tests.algorithm_validation_tests.exareme3.conftest import parse_response
from tests.algorithm_validation_tests.exareme3.helpers import get_test_params

algorithm_name = "pca"

expected_file = Path(__file__).parent / "expected" / f"{algorithm_name}_expected.json"


@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_pca_algorithm(test_input, expected):
    response = analysis_request(algorithm_name, test_input)
    result = parse_response(response)

    assert result
    assert result["title"]
    assert result["n_obs"] > 0
    assert result["variables"] == test_input["algorithm"]["y"]

    components = result["components"]
    assert components
    assert [component["component_id"] for component in components] == [
        f"PC{index}" for index in range(1, len(components) + 1)
    ]
    expected_component_fields = {
        "component_id",
        "explained_variance",
        "explained_variance_ratio",
        "cumulative_explained_variance",
        "loadings",
    }
    assert all(
        expected_component_fields <= component.keys() for component in components
    )

    recipe = result["reusable_preprocessing"]
    assert isinstance(recipe, dict)
    assert recipe
    assert recipe["base_parameters"]["variables"] == test_input["algorithm"]["y"]
    assert recipe["base_parameters"]["pca_variant"] == "pca"
