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
