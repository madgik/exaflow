def test_logistic_regression(get_algorithm_result):
    input = {
        "inputdata": {
            "data_model": "dementia:0.1",
            "datasets": [
                "1",
                "2",
                "3",
                "4",
                "6",
                "7",
                "18",
                "8",
                "9",
                "10",
            ],
            "validation_datasets": ["11"],
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
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "10",
            ],
            "validation_datasets": ["11"],
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
