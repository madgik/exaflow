def test_logistic_regression(get_algorithm_result):
    input = {
        "inputdata": {
            "data_model": "dementia:0.1",
            "datasets": [
                "PPMI_0",
                "PPMI_1",
                "PPMI_2",
                "PPMI_3",
                "PPMI_5",
                "PPMI_6",
                "EDSD_6",
                "PPMI_7",
                "PPMI_8",
                "PPMI_9",
            ],
            "validation_datasets": ["PPMI_TEST"],
            "filters": None,
            "variables": ["lefthippocampus", "gender"],
        },
        "test_case_num": 99,
        "algorithm": {
            "x": ["lefthippocampus"],
            "y": ["gender"],
            "parameters": None,
        },
    }
    algorithm_result = get_algorithm_result(
        "logistic_regression_fedaverage_flower", input
    )
    assert "accuracy" in algorithm_result


def test_logistic_regression_with_filters(get_algorithm_result):
    input = {
        "inputdata": {
            "data_model": "dementia:0.1",
            "datasets": [
                "PPMI_0",
                "PPMI_1",
                "PPMI_2",
                "PPMI_3",
                "PPMI_4",
                "PPMI_5",
                "PPMI_6",
                "PPMI_7",
                "PPMI_8",
                "PPMI_9",
            ],
            "validation_datasets": ["PPMI_TEST"],
            "filters": {
                "condition": "AND",
                "rules": [
                    {
                        "id": "lefthippocampus",
                        "field": "lefthippocampus",
                        "type": "double",
                        "input": "number",
                        "operator": "greater",
                        "value": 3.2,
                    },
                ],
                "valid": True,
            },
            "variables": ["lefthippocampus", "gender"],
        },
        "test_case_num": 99,
        "algorithm": {
            "x": ["lefthippocampus"],
            "y": ["gender"],
            "parameters": None,
        },
    }
    algorithm_result = get_algorithm_result(
        "logistic_regression_fedaverage_flower", input
    )
    assert "accuracy" in algorithm_result
