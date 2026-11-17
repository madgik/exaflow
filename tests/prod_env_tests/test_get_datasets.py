import json

import pytest
import requests

from tests.prod_env_tests import datasets_url


@pytest.fixture
def expected_datasets_per_data_model():
    return {
        "dementia:0.1": {
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
            "21",
            "22",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "23",
            "24",
            "25",
            "26",
            "27",
            "28",
            "29",
            "30",
            "31",
            "32",
            "33",
        },
        "tbi:0.1": {
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
        },
        "longitudinal_dementia:0.1": {
            "1",
            "2",
            "3",
            "4",
        },
    }


def test_get_datasets(expected_datasets_per_data_model):
    request = requests.get(datasets_url)
    response = json.loads(request.text)
    datasets_per_data_model = {}
    for data_model, datasets in response.items():
        if data_model not in datasets_per_data_model.keys():
            datasets_per_data_model[data_model] = set(datasets)
        else:
            datasets_per_data_model[data_model].update(datasets)

    assert datasets_per_data_model == expected_datasets_per_data_model
