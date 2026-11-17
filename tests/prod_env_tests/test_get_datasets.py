import json

import pytest
import requests

from tests.prod_env_tests import datasets_url


@pytest.fixture
def expected_datasets_per_data_model():
    return {
        "dementia:0.1": {
            "EDSD_0",
            "EDSD_1",
            "EDSD_2",
            "EDSD_3",
            "EDSD_4",
            "EDSD_5",
            "EDSD_6",
            "EDSD_7",
            "EDSD_8",
            "EDSD_9",
            "EDSD_TEST",
            "PPMI_0",
            "PPMI_1",
            "PPMI_2",
            "PPMI_3",
            "PPMI_4",
            "PPMI_5",
            "PPMI_6",
            "PPMI_7",
            "PPMI_8",
            "PPMI_9",
            "PPMI_TEST",
            "DESD-synthdata_0",
            "DESD-synthdata_1",
            "DESD-synthdata_2",
            "DESD-synthdata_3",
            "DESD-synthdata_4",
            "DESD-synthdata_5",
            "DESD-synthdata_6",
            "DESD-synthdata_7",
            "DESD-synthdata_8",
            "DESD-synthdata_9",
            "DESD-synthdata_TEST",
        },
        "tbi:0.1": {
            "Dummy TBI_0",
            "Dummy TBI_1",
            "Dummy TBI_2",
            "Dummy TBI_3",
            "Dummy TBI_4",
            "Dummy TBI_5",
            "Dummy TBI_6",
            "Dummy TBI_7",
            "Dummy TBI_8",
            "Dummy TBI_9",
            "Dummy TBI test",
        },
        "longitudinal_dementia:0.1": {
            "longitudinal_dementia0",
            "longitudinal_dementia1",
            "longitudinal_dementia2",
            "longitudinal_dementia_test",
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
