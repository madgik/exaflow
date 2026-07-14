from tests.algorithm_validation_tests.exareme3.conftest import analysis_request
from tests.algorithm_validation_tests.exareme3.conftest import parse_response


def test_kmeans_cluster_creator_full_output_feeds_chi_squared():
    payload = {
        "inputdata": {
            "data_model": "dementia:0.1",
            "datasets": [
                "edsd0",
                "edsd1",
                "edsd2",
                "edsd3",
                "edsd4",
                "edsd5",
                "edsd6",
                "edsd7",
                "edsd8",
                "edsd9",
            ],
            "filters": None,
            "variables": [
                "lefthippocampus",
                "righthippocampus",
                "gender",
            ],
        },
        "preprocessing": [
            {
                "name": "missing_values_handler",
                "parameters": {
                    "strategies": {
                        "lefthippocampus": "drop",
                        "righthippocampus": "drop",
                        "gender": "drop",
                    }
                },
            },
            {
                "name": "kmeans_cluster_creator",
                "parameters": {
                    "code": "kmeans_cluster",
                    "cluster_variables": [
                        "lefthippocampus",
                        "righthippocampus",
                    ],
                    "k_selection": "manual",
                    "k": 2,
                    "output_mode": "full",
                    "tol": 0.0001,
                    "maxiter": 100,
                },
            },
        ],
        "algorithm": {
            "x": ["kmeans_cluster"],
            "y": ["gender"],
            "parameters": {},
        },
    }

    response = analysis_request("chi_squared", payload, drop_na=False)
    result = parse_response(response)

    assert "chi2" in result
    assert "p_value" in result
    assert "dof" in result
    assert "expected" in result
