import json

import pytest
import requests

from tests.prod_env_tests import algorithm_specifications_url
from tests.prod_env_tests import preprocessing_specifications_url


def test_get_algorithm_specifications():
    request = requests.get(algorithm_specifications_url)
    result = json.loads(request.text)
    assert len(result) > 0


def test_get_logistic_regression():
    request = requests.get(algorithm_specifications_url)
    algorithms = json.loads(request.text)
    algorithm_names = [algorithm["name"] for algorithm in algorithms]
    assert "logistic_regression" in algorithm_names


def test_get_cox_regression_stacked():
    request = requests.get(algorithm_specifications_url)
    algorithms = json.loads(request.text)
    algorithm_names = [algorithm["name"] for algorithm in algorithms]
    assert "cox_regression_stacked" in algorithm_names


def test_get_cox_regression_classical():
    request = requests.get(algorithm_specifications_url)
    algorithms = json.loads(request.text)
    algorithm_names = [algorithm["name"] for algorithm in algorithms]
    assert "cox_regression_classical" in algorithm_names


def test_preprocessing_specifications_include_longitudinal_transformer():
    request = requests.get(preprocessing_specifications_url)
    result = json.loads(request.text)

    preprocessing_names = [preprocessing["name"] for preprocessing in result]
    assert "longitudinal_transformer" in preprocessing_names


def test_preprocessing_specifications_include_kmeans_cluster_creator():
    request = requests.get(preprocessing_specifications_url)
    result = json.loads(request.text)

    kmeans_spec = next(
        preprocessing
        for preprocessing in result
        if preprocessing["name"] == "kmeans_cluster_creator"
    )
    parameter_names = set(kmeans_spec["parameters"])

    assert parameter_names == {
        "new_column_names",
        "k",
        "enumeration_prefix",
        "reusable_preprocessing",
    }
    assert kmeans_spec["outputs"] == {
        "new_column_names_param": "new_column_names",
        "columns": [
            {
                "type": "text",
                "stattype": "nominal",
                "enumerations": {
                    "source_parameter_name": "k",
                    "source_type": "prefixed_range",
                    "prefix_parameter_name": "enumeration_prefix",
                },
            }
        ],
    }
