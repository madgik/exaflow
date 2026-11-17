from tests.algorithm_validation_tests.exareme3.conftest import analysis_request
from tests.algorithm_validation_tests.exareme3.conftest import parse_response

algorithm_name = "linear_svm"


def test_linear_svm_properties():
    test_input = {
        "inputdata": {
            "variables": ["rightgregyrusrectus", "dataset"],
            "data_model": "dementia:0.1",
            "datasets": [
                "3",
                "29",
                "7",
                "23",
                "9",
                "10",
                "5",
                "27",
                "18",
                "14",
                "20",
                "28",
                "26",
                "25",
                "13",
                "1",
                "17",
                "6",
                "2",
            ],
            "filters": None,
        },
        "algorithm": {
            "x": ["rightgregyrusrectus"],
            "y": ["dataset"],
            "parameters": {"gamma": 0.2, "C": 0.90},
        },
        "test_case_num": 0,
    }
    response = analysis_request(algorithm_name, test_input)
    result = parse_response(response)

    assert isinstance(result, dict)
    assert "title" in result
    assert "n_obs" in result
    assert "weights" in result
    assert "intercept" in result
    assert isinstance(result["weights"], list)
