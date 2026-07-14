import json
from pathlib import Path

import pytest

from tests.algorithm_validation_tests.exareme3.helpers import analysis_request

input_file = Path(__file__).parent / "input" / "kmeans_input.json"


def get_test_inputs():
    with input_file.open() as f:
        return json.load(f)["test_cases"]


@pytest.mark.parametrize("test_input", get_test_inputs())
def test_kmeans(test_input):
    response = analysis_request("kmeans", test_input)
    try:
        result = json.loads(response.text)
    except json.decoder.JSONDecodeError:
        raise ValueError(f"The result is not valid json:\n{response.text}") from None

    assert result["title"] == "K-Means Cluster Report"
    assert result["result_type"] == "privacy_safe_cluster_report"
    assert result["k_selection"] == "manual"
    assert result["selected_k"] == test_input["algorithm"]["parameters"]["k"]
    assert result["initialization_method"] == "random_range"
    assert result["n_init"] == 1
    assert result["selected_initialization"] == 0
    assert result["n_obs_interval"]
    assert "not a real patient" in result["center_definition"]
    assert result["privacy_note"]
    assert result["limitations"]
    assert len(result["clusters"]) == result["selected_k"]
    assert "centers" not in result
    for cluster in result["clusters"]:
        assert set(cluster) == {
            "cluster_id",
            "label",
            "size_interval",
            "center",
            "profile",
            "interpretation",
            "quality",
        }
