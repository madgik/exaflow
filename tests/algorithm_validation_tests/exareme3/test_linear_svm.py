from tests.algorithm_validation_tests.exareme3.helpers import algorithm_request
from tests.algorithm_validation_tests.exareme3.helpers import parse_response

algorithm_name = "linear_svm"


def test_linear_svm_properties():
    test_input = {
        "inputdata": {
            "y": ["dataset"],
            "x": ["rightgregyrusrectus"],
            "data_model": "dementia:0.1",
            "datasets": [
                "ppmi2",
                "desd-synthdata6",
                "ppmi6",
                "desd-synthdata0",
                "ppmi8",
                "ppmi9",
                "ppmi4",
                "desd-synthdata4",
                "edsd6",
                "edsd2",
                "edsd8",
                "desd-synthdata5",
                "desd-synthdata3",
                "desd-synthdata2",
                "edsd1",
                "ppmi0",
                "edsd5",
                "ppmi5",
                "ppmi1",
            ],
            "filters": None,
        },
        "parameters": {"gamma": 0.2, "C": 0.90},
        "test_case_num": 0,
    }
    response = algorithm_request(algorithm_name, test_input)
    result = parse_response(response)

    assert isinstance(result, dict)
    assert "title" in result
    assert "n_obs" in result
    assert "weights" in result
    assert "intercept" in result
    assert isinstance(result["weights"], list)
