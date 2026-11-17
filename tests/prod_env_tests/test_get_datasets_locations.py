import json

import pytest
import requests

from tests.prod_env_tests import datasets_locations_url


@pytest.fixture
def expected_datasets_locations():
    return {
        "dementia:0.1": {
            "DESD-synthdata_0": "localworker3",
            "DESD-synthdata_1": "localworker2",
            "DESD-synthdata_2": "localworker1",
            "DESD-synthdata_3": "localworker2",
            "DESD-synthdata_4": "localworker1",
            "DESD-synthdata_5": "localworker2",
            "DESD-synthdata_6": "localworker1",
            "DESD-synthdata_7": "localworker2",
            "DESD-synthdata_8": "localworker1",
            "DESD-synthdata_9": "localworker2",
            "DESD-synthdata_TEST": "globalworker",
            "EDSD_0": "localworker3",
            "EDSD_1": "localworker1",
            "EDSD_2": "localworker2",
            "EDSD_3": "localworker1",
            "EDSD_4": "localworker2",
            "EDSD_5": "localworker1",
            "EDSD_6": "localworker2",
            "EDSD_7": "localworker1",
            "EDSD_8": "localworker2",
            "EDSD_9": "localworker1",
            "EDSD_TEST": "globalworker",
            "PPMI_0": "localworker3",
            "PPMI_1": "localworker2",
            "PPMI_2": "localworker1",
            "PPMI_3": "localworker2",
            "PPMI_4": "localworker1",
            "PPMI_5": "localworker2",
            "PPMI_6": "localworker1",
            "PPMI_7": "localworker2",
            "PPMI_8": "localworker1",
            "PPMI_9": "localworker2",
            "PPMI_TEST": "globalworker",
        },
        "longitudinal_dementia:0.1": {
            "longitudinal_dementia0": "localworker3",
            "longitudinal_dementia1": "localworker2",
            "longitudinal_dementia2": "localworker1",
            "longitudinal_dementia_test": "globalworker",
        },
        "tbi:0.1": {
            "Dummy TBI_0": "localworker3",
            "Dummy TBI_1": "localworker2",
            "Dummy TBI_2": "localworker1",
            "Dummy TBI_3": "localworker2",
            "Dummy TBI_4": "localworker1",
            "Dummy TBI_5": "localworker2",
            "Dummy TBI_6": "localworker1",
            "Dummy TBI_7": "localworker2",
            "Dummy TBI_8": "localworker1",
            "Dummy TBI_9": "localworker2",
            "Dummy TBI test": "globalworker",
        },
    }


def test_get_dataset_location(expected_datasets_locations):
    request = requests.get(datasets_locations_url)
    response = json.loads(request.text)
    assert response == expected_datasets_locations
