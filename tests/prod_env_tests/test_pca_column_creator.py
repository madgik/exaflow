from copy import deepcopy
from pathlib import Path

from tests.algorithm_validation_tests.exareme3.conftest import analysis_request
from tests.algorithm_validation_tests.exareme3.conftest import parse_response
from tests.algorithm_validation_tests.exareme3.helpers import get_test_params

LOGISTIC_EXPECTED_FILE = (
    Path(__file__).parent / "expected" / "logistic_regression_expected.json"
)
PCA_VARIABLES = [
    "rightttgtransversetemporalgyrus",
    "leftpinsposteriorinsula",
]
OUTCOME_VARIABLE = "alzheimerbroadcategory"
POSITIVE_CLASS = "Other"


def test_pca_reporter_recipe_creator_feeds_logistic_regression():
    logistic_input, _ = get_test_params(LOGISTIC_EXPECTED_FILE)[0]

    reporter_payload = deepcopy(logistic_input)
    reporter_payload["preprocessing"] = [
        {
            "name": "missing_values_handler",
            "parameters": {
                "strategies": {variable: "drop" for variable in PCA_VARIABLES}
            },
        }
    ]
    reporter_payload["algorithm"] = {
        "x": None,
        "y": PCA_VARIABLES,
        "parameters": {},
    }

    reporter_response = analysis_request("pca", reporter_payload, drop_na=False)
    reporter_result = parse_response(reporter_response)

    recipe = reporter_result["reusable_preprocessing"]
    assert isinstance(recipe, dict)
    assert recipe
    assert recipe["base_parameters"]["variables"] == PCA_VARIABLES
    assert recipe["base_parameters"]["pca_variant"] == "pca"

    creator_payload = deepcopy(logistic_input)
    creator_payload["preprocessing"] = [
        {
            "name": "missing_values_handler",
            "parameters": {
                "strategies": {
                    **{variable: "drop" for variable in PCA_VARIABLES},
                    OUTCOME_VARIABLE: "drop",
                }
            },
        },
        {
            "name": "pca_column_creator",
            "parameters": {
                "reusable_preprocessing": recipe,
                "selected_components": ["PC1"],
                "code_prefix": "pca",
            },
        },
    ]
    creator_payload["algorithm"] = {
        "x": ["pca_PC1"],
        "y": [OUTCOME_VARIABLE],
        "parameters": {"positive_class": POSITIVE_CLASS},
    }

    logistic_response = analysis_request(
        "logistic_regression", creator_payload, drop_na=False
    )
    logistic_result = parse_response(logistic_response)

    assert logistic_result["dependent_var"] == OUTCOME_VARIABLE
    assert logistic_result["indep_vars"] == ["Intercept", "pca_PC1"]
    assert logistic_result["summary"]["n_obs"] > 0
    assert len(logistic_result["summary"]["coefficients"]) == len(
        logistic_result["indep_vars"]
    )
