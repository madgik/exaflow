"""HTTP coverage of ordered multi-column StandardScaler preprocessing."""

import csv
import math
from collections import Counter
from pathlib import Path

import pytest
import requests

from tests.prod_env_tests import analysis_url
from tests.prod_env_tests import preprocessing_specifications_url

SOURCE = "lefthippocampus"
OUTPUT = "hippocampus__scaled"


def payload(preprocessing, variables, *, datasets=None, source_variables=None):
    return {
        "inputdata": {
            "data_model": "dementia:0.1",
            "datasets": datasets
            if datasets is not None
            else ["edsd0", "edsd1", "edsd2", "edsd3"],
            "filters": None,
            "variables": [SOURCE] if source_variables is None else source_variables,
        },
        "preprocessing": preprocessing,
        "algorithm": {"name": "describe", "x": [], "y": variables, "parameters": {}},
    }


def step(**overrides):
    return {
        "name": "standard_scaler",
        "parameters": {
            "source_variables": [SOURCE],
            "new_column_names": [OUTPUT],
            **overrides,
        },
    }


def global_stats(result, variable):
    records = [
        record
        for record in result["featurewise"]
        if record["variable"] == variable and record["dataset"] == "all datasets"
    ]
    assert len(records) == 1
    assert records[0]["data"] is not None
    return records[0]["data"]


def test_standard_scaler_public_specification():
    response = requests.get(preprocessing_specifications_url, timeout=10)
    response.raise_for_status()
    specification = next(
        item for item in response.json() if item["name"] == "standard_scaler"
    )
    assert set(specification["parameters"]) == {"source_variables", "new_column_names"}
    assert specification["parameters"]["source_variables"]["multiple"]
    assert specification["outputs"]["new_column_names_param"] == "new_column_names"
    assert len(specification["outputs"]["columns"]) == 1
    assert specification["outputs"]["columns"][0]["type"] == "real"
    assert (
        specification["outputs"]["columns"][0]["column_count_param"]
        == "source_variables"
    )


@pytest.mark.parametrize("datasets", [["edsd0"], ["edsd0", "edsd1", "edsd2", "edsd3"]])
@pytest.mark.parametrize(
    "sources",
    [
        [SOURCE],
        ["righthippocampus", SOURCE],
        ["leftamygdala", SOURCE, "righthippocampus"],
    ],
)
def test_standard_scaler_source_preserved_and_output_visible_downstream(
    datasets, sources
):
    # The single-dataset case leaves other selected participants without rows.
    names = [f"derived_{index}" for index in range(len(sources))]
    baseline = requests.post(
        analysis_url,
        json=payload([], sources, datasets=datasets, source_variables=sources),
        timeout=(5, 120),
    )
    assert baseline.status_code == 200, baseline.text
    response = requests.post(
        analysis_url,
        json=payload(
            [step(source_variables=sources, new_column_names=names)],
            sources + names,
            datasets=datasets,
            source_variables=sources,
        ),
        timeout=(5, 120),
    )
    assert response.status_code == 200, response.text
    for variable, name in zip(sources, names):
        expected = global_stats(baseline.json(), variable)
        source = global_stats(response.json(), variable)
        output = global_stats(response.json(), name)
        for key in ("num_dtps", "num_na", "num_total", "mean", "std", "min", "max"):
            assert source[key] == pytest.approx(expected[key])
        n = expected["num_dtps"]
        assert n > 1
        population_scale = expected["std"] * math.sqrt((n - 1) / n)
        assert population_scale > 0
        assert output["num_dtps"] == n
        assert output["num_na"] == expected["num_na"]
        assert output["mean"] == pytest.approx(0, abs=1e-10)
        assert output["std"] == pytest.approx(math.sqrt(n / (n - 1)), rel=1e-8)
        for key in ("min", "max"):
            assert output[key] == pytest.approx(
                (expected[key] - expected["mean"]) / population_scale,
                rel=1e-8,
                abs=1e-10,
            )


@pytest.mark.parametrize("case", ["small_group", "empty_population"])
def test_standard_scaler_existing_masking_and_next_request_succeeds(case):
    # Select one observed fixture value using midpoint bounds, avoiding fragile
    # equality comparisons between CSV and SQL floating-point parsers.
    fixture = Path(__file__).parents[1] / "test_data" / "dementia_v_0_1" / "edsd0.csv"
    with fixture.open(newline="", encoding="utf-8") as stream:
        counts = Counter(
            float(row[SOURCE]) for row in csv.DictReader(stream) if row[SOURCE]
        )
    values = sorted(value for value in counts if math.isfinite(value))
    index = next(i for i in range(1, len(values) - 1) if counts[values[i]] == 1)
    lower = (values[index - 1] + values[index]) / 2
    upper = (values[index] + values[index + 1]) / 2
    if case == "empty_population":
        lower, upper = values[-1] + 1, values[-1] + 2
    request = payload([step()], [SOURCE, OUTPUT], datasets=["edsd0"])
    request["inputdata"]["filters"] = {
        "condition": "AND",
        "rules": [
            {
                "id": SOURCE,
                "field": SOURCE,
                "type": "real",
                "input": "number",
                "operator": "between",
                "value": [lower, upper],
            }
        ],
    }
    response = requests.post(analysis_url, json=request, timeout=(5, 60))
    if case == "empty_population":
        assert response.status_code == 460, response.text
        assert "no valid observations" in response.text, response.text
    else:
        # Describe retains its own masking; the scaler has no extra group floor.
        assert response.status_code == 200, response.text
        records = [
            record
            for record in response.json()["featurewise"]
            if record["variable"] in {SOURCE, OUTPUT}
        ]
        assert {record["variable"] for record in records} == {SOURCE, OUTPUT}
        assert all(record["data"] is None for record in records)
    # A failed empty fit must also release peers waiting in aggregation.
    following = requests.post(
        analysis_url,
        json=payload([step()], [SOURCE, OUTPUT], datasets=["edsd0"]),
        timeout=(5, 120),
    )
    assert following.status_code == 200, following.text
    assert global_stats(following.json(), OUTPUT)["num_dtps"] > 1


@pytest.mark.parametrize(
    "overrides,expected_error",
    [
        (
            {"new_column_names": ["a", "b"]},
            "'new_column_names' must contain exactly 1",
        ),
        (
            {"new_column_names": [SOURCE]},
            f"cannot create CDE '{SOURCE}' because it already exists",
        ),
        (
            {"recipe": {}},
            "Parameter recipe does not exist in the algorithm specification.",
        ),
        (
            {"minimum_row_count": 1},
            "Parameter minimum_row_count does not exist in the algorithm specification.",
        ),
        (
            {"source_variable": SOURCE},
            "Parameter source_variable does not exist in the algorithm specification.",
        ),
        (
            {"n_variables": 1},
            "Parameter n_variables does not exist in the algorithm specification.",
        ),
    ],
)
def test_standard_scaler_invalid_contract_rejected(overrides, expected_error):
    response = requests.post(
        analysis_url, json=payload([step(**overrides)], [SOURCE]), timeout=(5, 30)
    )
    assert response.status_code == 460, response.text
    assert expected_error in response.text, response.text


@pytest.mark.parametrize("names", [["one"], ["one", "two", "three"]])
def test_standard_scaler_multiple_source_output_count_mismatch(names):
    sources = [SOURCE, "righthippocampus"]
    response = requests.post(
        analysis_url,
        json=payload(
            [step(source_variables=sources, new_column_names=names)],
            sources,
            source_variables=sources,
        ),
        timeout=(5, 30),
    )
    assert response.status_code == 460, response.text
    assert "must contain exactly 2" in response.text, response.text
