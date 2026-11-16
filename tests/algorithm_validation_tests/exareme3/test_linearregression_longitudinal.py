from tests.algorithm_validation_tests.exareme3.helpers import analysis_request
from tests.algorithm_validation_tests.exareme3.helpers import parse_response

algorithm_name = "linear_regression"


def make_test_input(visit1: str, visit2: str) -> dict:
    return {
        "inputdata": {
            "data_model": "longitudinal_dementia:0.1",
            "variables": ["righthippocampus", "agegroup", "gender", "lefthippocampus"],
            "datasets": [
                "longitudinal_dementia0",
                "longitudinal_dementia1",
                "longitudinal_dementia2",
            ],
            "filters": None,
        },
        "preprocessing": [
            {
                "name": "longitudinal_transformer",
                "parameters": {
                    "visit1": visit1,
                    "visit2": visit2,
                    "strategies": {
                        "lefthippocampus": "diff",
                        "righthippocampus": "diff",
                        "agegroup": "second",
                        "gender": "first",
                    },
                },
            },
        ],
        "algorithm": {
            "x": ["righthippocampus", "agegroup", "gender"],
            "y": ["lefthippocampus"],
            "parameters": None,
        },
    }


def test_linearregression_algorithm_27nobs():
    test_input_27nobs = make_test_input(visit1="1", visit2="5")
    response = analysis_request(algorithm_name, test_input_27nobs)
    result = parse_response(response)

    assert result["n_obs"] == 27
    assert result["dependent_var"] == "lefthippocampus"
    assert result["indep_vars"] == [
        "Intercept",
        "gender[1]",
        "agegroup",
        "righthippocampus",
    ]


def test_linearregression_algorithm_81nobs():
    test_input_81nobs = make_test_input(visit1="1", visit2="2")
    response = analysis_request(algorithm_name, test_input_81nobs)
    result = parse_response(response)

    assert result["n_obs"] == 81
    assert result["dependent_var"] == "lefthippocampus"
    assert result["indep_vars"] == [
        "Intercept",
        "gender[1]",
        "agegroup",
        "righthippocampus",
    ]
