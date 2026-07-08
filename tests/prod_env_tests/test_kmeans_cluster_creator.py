from tests.algorithm_validation_tests.exareme3.conftest import analysis_request
from tests.algorithm_validation_tests.exareme3.conftest import parse_response


def _inputdata():
    return {
        "data_model": "dementia:0.1",
        "datasets": [f"edsd{index}" for index in range(10)],
        "filters": None,
        "variables": ["lefthippocampus", "righthippocampus", "gender"],
    }


def test_kmeans_cluster_creator_full_output_feeds_chi_squared():
    cluster_variables = ["lefthippocampus", "righthippocampus"]
    report_payload = {
        "inputdata": {**_inputdata(), "variables": cluster_variables},
        "preprocessing": [
            {
                "name": "missing_values_handler",
                "parameters": {
                    "strategies": {variable: "drop" for variable in cluster_variables}
                },
            }
        ],
        "algorithm": {
            "x": None,
            "y": cluster_variables,
            "parameters": {
                "k_selection": "manual",
                "k": 2,
                "tol": 0.0001,
                "maxiter": 100,
            },
        },
    }
    report = parse_response(analysis_request("kmeans", report_payload, drop_na=False))
    reusable = report["reusable_preprocessing"]

    payload = {
        "inputdata": _inputdata(),
        "preprocessing": [
            {
                "name": "missing_values_handler",
                "parameters": {
                    "strategies": {
                        variable: "drop" for variable in _inputdata()["variables"]
                    }
                },
            },
            {
                "name": "kmeans_cluster_creator",
                "parameters": {
                    "code": "kmeans_cluster",
                    "reusable_preprocessing": reusable,
                },
            },
        ],
        "algorithm": {
            "x": ["kmeans_cluster"],
            "y": ["gender"],
            "parameters": {},
        },
    }

    result = parse_response(analysis_request("chi_squared", payload, drop_na=False))

    assert {"chi2", "p_value", "dof", "expected"} <= set(result)
