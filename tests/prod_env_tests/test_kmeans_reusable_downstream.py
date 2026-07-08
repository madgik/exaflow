import json
import math

import pytest

from exaflow.algorithms.exareme3.cluster.kmeans import KMeansReusablePreprocessing
from exaflow.algorithms.exareme3.cluster.kmeans import (
    build_kmeans_preprocessing_parameters,
)
from tests.algorithm_validation_tests.exareme3.conftest import analysis_request
from tests.algorithm_validation_tests.exareme3.conftest import parse_response

DATA_MODEL = "dementia:0.1"
DATASETS = [f"edsd{index}" for index in range(10)]
KMEANS_VARIABLES = ["lefthippocampus", "righthippocampus"]
GENERATED_CODE = "kmeans_cluster"


def _inputdata(variables):
    return {
        "data_model": DATA_MODEL,
        "datasets": DATASETS,
        "filters": None,
        "variables": list(dict.fromkeys(variables)),
    }


def _missing_values_step(variables):
    return {
        "name": "missing_values_handler",
        "parameters": {"strategies": {variable: "drop" for variable in variables}},
    }


@pytest.fixture(scope="module")
def creator_parameters():
    reporter_payload = {
        "inputdata": _inputdata(KMEANS_VARIABLES),
        "preprocessing": [_missing_values_step(KMEANS_VARIABLES)],
        "algorithm": {
            "x": None,
            "y": KMEANS_VARIABLES,
            "parameters": {
                "k_selection": "manual",
                "k": 3,
                "tol": 0.0001,
                "maxiter": 100,
            },
        },
    }
    report = parse_response(analysis_request("kmeans", reporter_payload, drop_na=False))
    reusable = KMeansReusablePreprocessing.model_validate(
        json.loads(json.dumps(report["reusable_preprocessing"]))
    )
    assert list(reusable.centers) == ["cluster_0", "cluster_1", "cluster_2"]
    assert [output.output_mode for output in reusable.available_outputs] == ["full"]
    return build_kmeans_preprocessing_parameters(reusable, code=GENERATED_CODE)


def test_saved_centers_replay_into_logistic_regression(creator_parameters):
    variables = KMEANS_VARIABLES + ["gender"]
    payload = {
        "inputdata": _inputdata(variables),
        "preprocessing": [
            _missing_values_step(variables),
            {"name": "kmeans_cluster_creator", "parameters": creator_parameters},
        ],
        "algorithm": {
            "x": [GENERATED_CODE],
            "y": ["gender"],
            "parameters": {"positive_class": "F"},
        },
    }

    result = parse_response(
        analysis_request("logistic_regression", payload, drop_na=False)
    )

    assert result["dependent_var"] == "gender"
    assert any(GENERATED_CODE in name for name in result["indep_vars"])
    for value in result["summary"]["coefficients"]:
        assert math.isfinite(float(value))
