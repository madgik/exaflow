from tests.algorithm_validation_tests.exareme3.helpers import algorithm_request
from tests.algorithm_validation_tests.exareme3.helpers import parse_response

algorithm_name = "linear_regression"


def make_test_input(visit1: str, visit2: str) -> dict:
    return {
        "inputdata": {
            "data_model": "longitudinal_dementia:0.1",
            "y": ["lefthippocampus"],
            "x": ["righthippocampus", "agegroup", "gender"],
            "datasets": [
                "longitudinal_dementia0",
                "longitudinal_dementia1",
                "longitudinal_dementia2",
            ],
            "filters": None,
        },
        "preprocessing": {
            "longitudinal_transformer": {
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
    }


def test_linearregression_algorithm_27nobs():
    test_input_27nobs = make_test_input(visit1="BL", visit2="FL4")
    response = algorithm_request(algorithm_name, test_input_27nobs)
    result = parse_response(response)

    assert result["n_obs"] == 27
    assert result["dependent_var"] == "lefthippocampus_diff"
    assert result["indep_vars"] == [
        "Intercept",
        "agegroup[50-59y]",
        "agegroup[60-69y]",
        "agegroup[70-79y]",
        "gender[M]",
        "righthippocampus_diff",
    ]


def test_linearregression_algorithm_81nobs():
    test_input_81nobs = make_test_input(visit1="BL", visit2="FL1")
    response = algorithm_request(algorithm_name, test_input_81nobs)
    result = parse_response(response)

    assert result["n_obs"] == 81
    assert result["dependent_var"] == "lefthippocampus_diff"
    assert result["indep_vars"] == [
        "Intercept",
        "agegroup[50-59y]",
        "agegroup[60-69y]",
        "agegroup[70-79y]",
        "gender[M]",
        "righthippocampus_diff",
    ]
