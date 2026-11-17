from tests.algorithm_validation_tests.exareme3.conftest import analysis_request
from tests.algorithm_validation_tests.exareme3.conftest import parse_response

algorithm_name = "linear_svm"


def test_linear_svm_properties():
    test_input = {
        "inputdata": {
            "variables": ["rightgregyrusrectus", "dataset"],
            "data_model": "dementia:0.1",
            "datasets": [
                "PPMI_2",
                "DESD-synthdata_6",
                "PPMI_6",
                "DESD-synthdata_0",
                "PPMI_8",
                "PPMI_9",
                "PPMI_4",
                "DESD-synthdata_4",
                "EDSD_6",
                "EDSD_2",
                "EDSD_8",
                "DESD-synthdata_5",
                "DESD-synthdata_3",
                "DESD-synthdata_2",
                "EDSD_1",
                "PPMI_0",
                "EDSD_5",
                "PPMI_5",
                "PPMI_1",
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
