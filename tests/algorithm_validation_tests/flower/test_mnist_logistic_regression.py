def test_mnist_logistic_regression(get_algorithm_result):
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
