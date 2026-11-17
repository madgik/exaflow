import json

import pytest
import requests

from tests.prod_env_tests import datasets_locations_url


@pytest.fixture
def expected_datasets_locations():
    return {
        "dementia:0.1": {
            "23": "localworker3",
            "24": "localworker2",
            "25": "localworker1",
            "26": "localworker2",
            "27": "localworker1",
            "28": "localworker2",
            "29": "localworker1",
            "30": "localworker2",
            "31": "localworker1",
            "32": "localworker2",
            "33": "globalworker",
            "12": "localworker3",
            "13": "localworker1",
            "14": "localworker2",
            "15": "localworker1",
            "16": "localworker2",
            "17": "localworker1",
            "18": "localworker2",
            "19": "localworker1",
            "20": "localworker2",
            "21": "localworker1",
            "22": "globalworker",
            "1": "localworker3",
            "2": "localworker2",
            "3": "localworker1",
            "4": "localworker2",
            "5": "localworker1",
            "6": "localworker2",
            "7": "localworker1",
            "8": "localworker2",
            "9": "localworker1",
            "10": "localworker2",
            "11": "globalworker",
        },
        "longitudinal_dementia:0.1": {
            "1": "localworker3",
            "2": "localworker2",
            "3": "localworker1",
            "4": "globalworker",
        },
        "tbi:0.1": {
            "1": "localworker3",
            "2": "localworker2",
            "3": "localworker1",
            "4": "localworker2",
            "5": "localworker1",
            "6": "localworker2",
            "7": "localworker1",
            "8": "localworker2",
            "9": "localworker1",
            "10": "localworker2",
            "11": "globalworker",
        },
    }


def test_get_dataset_location(expected_datasets_locations):
    request = requests.get(datasets_locations_url)
    response = json.loads(request.text)
    assert response == expected_datasets_locations
