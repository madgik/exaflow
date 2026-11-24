"""HTTP integration of source-preserving numerical transformations.

Requires current controller/worker images, aggregation enabled, and the standard
dementia fixtures used by the existing production suites. No longitudinal data
or longitudinal preprocessing is involved.
"""

import math
from copy import deepcopy

import pytest
import requests

from tests.prod_env_tests import analysis_url
from tests.prod_env_tests import preprocessing_specifications_url

SOURCE = "lefthippocampus"
OUTPUT = "hippocampus__prepared"
DATASETS = ["edsd0", "edsd1", "edsd2", "edsd3"]
STEP_NAMES = {
    "variable_scaler",
    "variable_transformer",
    "global_imputer",
    "missing_indicator",
    "global_winsorizer",
    "outlier_detector",
}


@pytest.fixture(scope="module", autouse=True)
def deployed_steps():
    response = requests.get(preprocessing_specifications_url, timeout=5)
    response.raise_for_status()
    steps = {item["name"]: item for item in response.json()}
    missing = STEP_NAMES - steps.keys()
    assert not missing, (
        f"Missing derived steps: {sorted(missing)}. Deploy the current "
        "transformations branch with the aggregation component enabled."
    )
    for name in STEP_NAMES:
        assert "source_variable" in steps[name]["parameters"]
        assert steps[name]["outputs"]["new_column_names_param"] == "new_column_names"
    return steps


def _step(name, **parameters):
    return {
        "name": name,
        "parameters": {
            "source_variable": SOURCE,
            "new_column_names": [OUTPUT],
            **parameters,
        },
    }


def _payload(steps, variables=None, algorithm="describe"):
    return {
        "inputdata": {
            "data_model": "dementia:0.1",
            "datasets": DATASETS,
            "filters": None,
            "variables": [SOURCE],
        },
        "preprocessing": steps,
        "algorithm": {
            "name": algorithm,
            "x": [],
            "y": variables or [SOURCE, OUTPUT],
            "parameters": {},
        },
    }


def _analysis(payload):
    response = requests.post(analysis_url, json=payload, timeout=(5, 120))
    assert response.status_code == 200, response.text
    return response.json()


def _global(result, variable):
    records = [
        record
        for record in result["featurewise"]
        if record["variable"] == variable and record["dataset"] == "all datasets"
    ]
    assert len(records) == 1
    assert records[0]["data"] is not None
    return records[0]["data"]


@pytest.fixture(scope="module")
def source_stats():
    result = _analysis(_payload([], variables=[SOURCE]))
    stats = _global(result, SOURCE)
    assert stats["num_dtps"] >= 40
    assert stats["std"] > 0
    return stats


def _assert_source_unchanged(result, baseline):
    actual = _global(result, SOURCE)
    for key in ("num_dtps", "num_na", "num_total"):
        assert actual[key] == baseline[key]
    for key in ("mean", "std", "min", "max"):
        assert actual[key] == pytest.approx(baseline[key], rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("name", sorted(STEP_NAMES))
def test_derived_step_specifications_are_exposed(deployed_steps, name):
    spec = deployed_steps[name]
    assert spec["parameters"]["source_variable"]["required"]
    assert spec["parameters"]["new_column_names"]["required"]
    assert len(spec["outputs"]["columns"]) == 1
    assert spec["outputs"]["columns"][0]["column_count_param"] is None


@pytest.mark.parametrize("method", ["standard", "robust", "minmax", "maxabs"])
def test_scalers_create_a_new_column_and_preserve_source(method, source_stats):
    result = _analysis(_payload([_step("variable_scaler", method=method)]))
    _assert_source_unchanged(result, source_stats)
    actual = _global(result, OUTPUT)
    assert actual["num_dtps"] == source_stats["num_dtps"]
    if method == "standard":
        assert actual["mean"] == pytest.approx(0, abs=1e-9)
        # Describe reports sample std; scaling uses population std.
        n = actual["num_dtps"]
        assert actual["std"] == pytest.approx(math.sqrt(n / (n - 1)), rel=1e-6)
    elif method == "minmax":
        assert actual["min"] == pytest.approx(0, abs=1e-9)
        assert actual["max"] == pytest.approx(1, abs=1e-9)
    elif method == "maxabs":
        maximum = max(abs(source_stats["min"]), abs(source_stats["max"]))
        assert actual["min"] == pytest.approx(source_stats["min"] / maximum)
        assert actual["max"] == pytest.approx(source_stats["max"] / maximum)
    else:
        iqr = source_stats["q3"] - source_stats["q1"]
        assert iqr > 0
        expected = (source_stats["mean"] - source_stats["median"]) / iqr
        assert actual["mean"] == pytest.approx(expected, abs=0.03)


@pytest.mark.parametrize(
    "method",
    ["log1p", "sqrt", "asinh", "yeo_johnson", "quantile_uniform", "quantile_normal"],
)
def test_nonlinear_derived_columns_feed_a_real_analysis(method, source_stats):
    parameters = {"method": method}
    if method.startswith("quantile_"):
        parameters["n_quantiles"] = 5
    result = _analysis(_payload([_step("variable_transformer", **parameters)]))
    _assert_source_unchanged(result, source_stats)
    actual = _global(result, OUTPUT)
    assert actual["num_dtps"] == source_stats["num_dtps"]
    assert math.isfinite(actual["mean"])
    if method in {"log1p", "sqrt", "asinh"}:
        function = {"log1p": math.log1p, "sqrt": math.sqrt, "asinh": math.asinh}[method]
        assert actual["min"] == pytest.approx(function(source_stats["min"]))
        assert actual["max"] == pytest.approx(function(source_stats["max"]))
    elif method == "quantile_uniform":
        assert 0 <= actual["min"] <= actual["max"] <= 1
    elif method == "quantile_normal":
        assert -5.2 <= actual["min"] <= actual["max"] <= 5.2


@pytest.mark.parametrize("strategy", ["mean", "median", "constant"])
def test_imputation_only_fills_the_derived_column(strategy, source_stats):
    parameters = {"strategy": strategy}
    if strategy == "constant":
        parameters["fill_value"] = -2.0
    result = _analysis(_payload([_step("global_imputer", **parameters)]))
    _assert_source_unchanged(result, source_stats)
    actual = _global(result, OUTPUT)
    assert actual["num_na"] == 0
    assert actual["num_dtps"] == source_stats["num_total"]
    replacement = {
        "mean": source_stats["mean"],
        "median": source_stats["median"],
        "constant": -2,
    }[strategy]
    expected = (
        source_stats["mean"] * source_stats["num_dtps"]
        + replacement * source_stats["num_na"]
    ) / source_stats["num_total"]
    assert actual["mean"] == pytest.approx(expected, abs=1e-5)


def test_missing_indicator_is_a_new_integer_column(source_stats):
    result = _analysis(_payload([_step("missing_indicator")]))
    _assert_source_unchanged(result, source_stats)
    actual = _global(result, OUTPUT)
    assert actual["num_dtps"] == source_stats["num_total"]
    assert actual["mean"] == pytest.approx(
        source_stats["num_na"] / source_stats["num_total"]
    )
    assert 0 <= actual["min"] <= actual["max"] <= 1


def test_global_winsorization_does_not_overwrite_source(source_stats):
    result = _analysis(
        _payload([_step("global_winsorizer", lower_quantile=0.25, upper_quantile=0.75)])
    )
    _assert_source_unchanged(result, source_stats)
    actual = _global(result, OUTPUT)
    tolerance = 0.01 * (source_stats["max"] - source_stats["min"])
    assert actual["min"] == pytest.approx(source_stats["q1"], abs=tolerance)
    assert actual["max"] == pytest.approx(source_stats["q3"], abs=tolerance)


@pytest.mark.parametrize("mode", ["zscore", "robust"])
def test_outlier_flags_do_not_delete_or_change_observations(mode, source_stats):
    result = _analysis(_payload([_step("outlier_detector", mode=mode, threshold=0.1)]))
    _assert_source_unchanged(result, source_stats)
    actual = _global(result, OUTPUT)
    assert actual["num_dtps"] == source_stats["num_total"]
    assert 0 <= actual["min"] <= actual["max"] <= 1


def test_standard_scaler_replays_an_explicit_named_recipe(source_stats):
    n = source_stats["num_dtps"]
    recipe = {
        "schema_version": "2",
        "primitive_name": "standard_scaler",
        "feature_names": [SOURCE],
        "categorical_features": [],
        "location": [source_stats["mean"]],
        "scale": [source_stats["std"] * math.sqrt((n - 1) / n)],
    }
    result = _analysis(
        _payload([_step("variable_scaler", method="standard", recipe=recipe)])
    )
    _assert_source_unchanged(result, source_stats)
    assert _global(result, OUTPUT)["mean"] == pytest.approx(0, abs=1e-9)


def test_derived_chain_keeps_every_source_available(source_stats):
    filled, logged = "hippocampus__filled", "hippocampus__logged"
    steps = [
        _step("global_imputer", strategy="mean", new_column_names=[filled]),
        _step(
            "variable_transformer",
            source_variable=filled,
            new_column_names=[logged],
            method="log1p",
        ),
        _step("variable_scaler", source_variable=logged, method="minmax"),
    ]
    result = _analysis(_payload(steps, variables=[SOURCE, filled, logged, OUTPUT]))
    _assert_source_unchanged(result, source_stats)
    for variable in (filled, logged, OUTPUT):
        assert _global(result, variable)["num_dtps"] == source_stats["num_total"]
    assert _global(result, OUTPUT)["min"] == pytest.approx(0, abs=1e-9)
    assert _global(result, OUTPUT)["max"] == pytest.approx(1, abs=1e-9)


def test_pca_can_select_derived_columns_without_reporter_preparation():
    scaled, logged = "hippocampus__scaled", "hippocampus__logged"
    steps = [
        _step("variable_scaler", new_column_names=[scaled], method="minmax"),
        _step("variable_transformer", new_column_names=[logged], method="log1p"),
    ]
    payload = _payload(steps, variables=[scaled, logged], algorithm="pca")
    # The existing source-only drop step runs before generated names exist.
    payload["preprocessing"].insert(
        0,
        {
            "name": "missing_values_handler",
            "parameters": {"strategies": {SOURCE: "drop"}},
        },
    )
    result = _analysis(payload)
    assert result["n_obs"] > 0
    assert result["reusable_preprocessing"]["base_parameters"]["variables"] == [
        scaled,
        logged,
    ]


@pytest.mark.parametrize("names", [[], ["a", "b"], [SOURCE], [" "]])
def test_invalid_derived_names_are_rejected_by_the_controller(names):
    payload = _payload([_step("variable_scaler", method="standard")])
    payload = deepcopy(payload)
    payload["preprocessing"][0]["parameters"]["new_column_names"] = names
    response = requests.post(analysis_url, json=payload, timeout=(5, 30))
    assert response.status_code == 460, response.text
