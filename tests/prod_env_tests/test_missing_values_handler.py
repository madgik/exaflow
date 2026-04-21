from copy import deepcopy

from tests.algorithm_validation_tests.exareme3.conftest import algorithm_request

base_logistic_input = {
    "inputdata": {
        "y": ["alzheimerbroadcategory"],
        "x": [
            "rightttgtransversetemporalgyrus",
            "leftpinsposteriorinsula",
            "leftpoparietaloperculum",
            "rightptplanumtemporale",
            "leftventraldc",
        ],
        "data_model": "dementia:0.1",
        "datasets": ["ppmi0", "desd-synthdata0"],
        "filters": None,
    },
    "parameters": {"positive_class": "Other"},
}


base_pca_input = {
    "inputdata": {
        "y": ["lefthippocampus"],
        "data_model": "dementia:0.1",
        "datasets": ["desd-synthdata8"],
        "filters": None,
    },
    "parameters": None,
}


base_describe_input = {
    "inputdata": {
        "y": [
            "subjectageyears",
            "rightpallidum",
            "ppmicategory",
            "adnicategory",
            "edsdcategory",
            "handedness",
            "dataset",
            "apoe4",
        ],
        "x": [],
        "data_model": "dementia:0.1",
        "datasets": [
            "edsd3",
            "edsd2",
            "edsd8",
            "desd-synthdata0",
            "ppmi8",
            "desd-synthdata5",
            "ppmi5",
            "desd-synthdata8",
            "desd-synthdata4",
            "edsd6",
        ],
        "filters": None,
    },
    "parameters": {},
}


def test_missing_values_handler_single_variable_median_strategy():
    payload = deepcopy(base_pca_input)
    payload["preprocessing"] = {
        "missing_values_handler": {"strategies": {"lefthippocampus": "median"}}
    }

    response = algorithm_request("pca", payload, drop_na=False)
    assert response.status_code == 200, f"{response.status_code}: {response.content}"


def test_missing_values_handler_per_variable_partial_strategies():
    payload = deepcopy(base_describe_input)
    payload["preprocessing"] = {
        "missing_values_handler": {
            "strategies": {
                "subjectageyears": "mean",
                "rightpallidum": "constant",
            },
            "fill_values": {"rightpallidum": 0.0},
        }
    }

    response = algorithm_request("describe", payload, drop_na=False)
    assert response.status_code == 200, f"{response.status_code}: {response.content}"


def test_missing_values_handler_rejects_fill_values_outside_categorical_enums():
    payload = deepcopy(base_describe_input)
    invalid_value = "__not_in_apoe4_enums__"
    payload["preprocessing"] = {
        "missing_values_handler": {
            "strategies": {"apoe4": "constant"},
            "fill_values": {"apoe4": invalid_value},
        }
    }

    response = algorithm_request("describe", payload, drop_na=False)
    assert response.status_code == 460, f"{response.status_code}: {response.content}"
    assert b"categorical enum codes" in response.content


def test_missing_values_handler_rejects_non_string_fill_value_for_categorical():
    payload = deepcopy(base_describe_input)
    payload["preprocessing"] = {
        "missing_values_handler": {
            "strategies": {"apoe4": "constant"},
            "fill_values": {"apoe4": 1},
        }
    }

    response = algorithm_request("describe", payload, drop_na=False)
    assert response.status_code == 460, f"{response.status_code}: {response.content}"
    assert b"text categorical enum code" in response.content


def test_missing_values_handler_rejects_mean_for_categorical_variable():
    payload = deepcopy(base_logistic_input)
    payload["preprocessing"] = {
        "missing_values_handler": {
            "strategies": {"alzheimerbroadcategory": "mean"},
        }
    }

    response = algorithm_request("logistic_regression", payload, drop_na=False)
    assert response.status_code == 460, f"{response.status_code}: {response.content}"
    assert b"can only be used for numerical variables" in response.content


def test_missing_values_handler_rejects_constant_categorical_without_fill_values():
    payload = deepcopy(base_logistic_input)
    payload["preprocessing"] = {
        "missing_values_handler": {
            "strategies": {"alzheimerbroadcategory": "constant"},
        }
    }

    response = algorithm_request("logistic_regression", payload, drop_na=False)
    assert response.status_code == 460, f"{response.status_code}: {response.content}"
    assert b"requires 'fill_values[alzheimerbroadcategory]'" in response.content


def test_missing_values_handler_rejects_fill_values_for_non_constant_strategy():
    payload = deepcopy(base_logistic_input)
    payload["preprocessing"] = {
        "missing_values_handler": {
            "strategies": {"rightttgtransversetemporalgyrus": "mean"},
            "fill_values": {"rightttgtransversetemporalgyrus": 0.0},
        }
    }

    response = algorithm_request("logistic_regression", payload, drop_na=False)
    assert response.status_code == 460, f"{response.status_code}: {response.content}"
    assert b"can only be provided when strategy is 'constant'" in response.content
