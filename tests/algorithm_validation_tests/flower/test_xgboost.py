def test_xgboost(get_algorithm_result):
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
    input["type"] = "flower"
    algorithm_result = get_algorithm_result("xgboost", input)
    # {'metrics_aggregated': {'AUC': 0.7575790087463558}}
    print(algorithm_result)
    auc_aggregated = algorithm_result["AUC"]
    auc_ascending = algorithm_result["auc_ascending"]

    assert auc_aggregated > 0.0
    assert auc_ascending == "correct"
