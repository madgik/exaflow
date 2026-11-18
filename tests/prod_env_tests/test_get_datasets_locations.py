import json

import pytest
import requests

from tests.prod_env_tests import datasets_locations_url


@pytest.fixture
def expected_datasets_locations():
    return {
        "dementia:0.1": {
            "desd-synthdata0": "localworker3",
            "desd-synthdata1": "localworker2",
            "desd-synthdata2": "localworker1",
            "desd-synthdata3": "localworker2",
            "desd-synthdata4": "localworker1",
            "desd-synthdata5": "localworker2",
            "desd-synthdata6": "localworker1",
            "desd-synthdata7": "localworker2",
            "desd-synthdata8": "localworker1",
            "desd-synthdata9": "localworker2",
            "desd-synthdata_test": "localworker3",
            "edsd0": "localworker3",
            "edsd1": "localworker1",
            "edsd2": "localworker2",
            "edsd3": "localworker1",
            "edsd4": "localworker2",
            "edsd5": "localworker1",
            "edsd6": "localworker2",
            "edsd7": "localworker1",
            "edsd8": "localworker2",
            "edsd9": "localworker1",
            "edsd_test": "localworker3",
            "ppmi0": "localworker3",
            "ppmi1": "localworker2",
            "ppmi2": "localworker1",
            "ppmi3": "localworker2",
            "ppmi4": "localworker1",
            "ppmi5": "localworker2",
            "ppmi6": "localworker1",
            "ppmi7": "localworker2",
            "ppmi8": "localworker1",
            "ppmi9": "localworker2",
            "ppmi_test": "localworker3",
        },
        "longitudinal_dementia:0.1": {
            "longitudinal_dementia0": "localworker3",
            "longitudinal_dementia1": "localworker2",
            "longitudinal_dementia2": "localworker1",
            "longitudinal_dementia_test": "localworker3",
        },
        "tbi:0.1": {
            "dummy_tbi0": "localworker3",
            "dummy_tbi1": "localworker2",
            "dummy_tbi2": "localworker1",
            "dummy_tbi3": "localworker2",
            "dummy_tbi4": "localworker1",
            "dummy_tbi5": "localworker2",
            "dummy_tbi6": "localworker1",
            "dummy_tbi7": "localworker2",
            "dummy_tbi8": "localworker1",
            "dummy_tbi9": "localworker2",
            "dummy_tbi_test": "localworker3",
        },
    }


def test_get_dataset_location(expected_datasets_locations):
    request = requests.get(datasets_locations_url)
    response = json.loads(request.text)
    assert response == expected_datasets_locations
