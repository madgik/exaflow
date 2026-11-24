import json
from pathlib import Path

import pytest

from tests.algorithm_validation_tests.exareme3.conftest import analysis_request
from tests.algorithm_validation_tests.exareme3.conftest import parse_response

input_file = Path(__file__).parent / "input" / "kmeans_input.json"


def get_test_inputs():
    with input_file.open() as f:
        return json.load(f)["test_cases"]


@pytest.mark.parametrize("test_input", get_test_inputs())
def test_kmeans(test_input):
    response = analysis_request("kmeans", test_input)
    result = parse_response(response)

    # this test only ensures that the algorithm runs smoothly without errors
    assert result


def test_kmeans_reporting_returns_privacy_safe_report():
    response = analysis_request("kmeans", get_test_inputs()[0])
    result = parse_response(response)

    assert result["result_type"] == "privacy_safe_cluster_report"
    assert result["k_selection"] == "manual"
    assert result["selected_k"] == 2
    assert result["center_definition"]
    assert result["privacy_note"]
    assert result["limitations"]
    assert len(result["clusters"]) == 2
    assert all("size_interval" in cluster for cluster in result["clusters"])


@pytest.mark.parametrize("k", [3, 4])
def test_manual_kmeans_rejects_small_clusters_without_exporting_centers(k):
    # These hippocampus fits contain a singleton with the fixed seed.
    payload = get_test_inputs()[0]
    payload["algorithm"]["parameters"]["k"] = k

    response = analysis_request("kmeans", payload)

    assert response.status_code == 460
    assert "minimum cohort size" in response.text
    assert '"clusters"' not in response.text
    assert '"reusable_preprocessing"' not in response.text


def test_kmeans_reporting_elbow_returns_selected_k_and_elbow_payload():
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
            "variables": ["lefthippocampus", "righthippocampus"],
        },
        "preprocessing": None,
        "algorithm": {
            "x": None,
            "y": ["lefthippocampus", "righthippocampus"],
            "parameters": {
                "k_selection": "elbow",
                "k_min": 2,
                "k_max": 4,
                "tol": 0.0001,
                "maxiter": 100,
            },
        },
    }

    response = analysis_request("kmeans", payload)
    result = parse_response(response)

    assert result["result_type"] == "privacy_safe_cluster_report"
    assert result["k_selection"] == "elbow"
    assert 2 <= result["selected_k"] <= 4
    assert result["elbow"]["k_min"] == 2
    assert result["elbow"]["k_max"] == 4
    assert result["elbow"]["selected_k"] == result["selected_k"]
    eligible_k = set(result["elbow"]["inertia_by_k"])
    assert str(result["selected_k"]) in eligible_k
    assert eligible_k <= {"2", "3", "4"}
    if eligible_k != {"2", "3", "4"}:
        assert "excluded" in result["elbow"]["warning"]
