def test_mnist_logistic_regression(get_algorithm_result):
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
    algorithm_result = get_algorithm_result("mnist_logistic_regression", input)
    assert {"accuracy": 0.8486} == algorithm_result
