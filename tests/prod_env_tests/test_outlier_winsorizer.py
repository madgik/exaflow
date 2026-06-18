from copy import deepcopy

import pytest

from tests.algorithm_validation_tests.exareme3.conftest import analysis_request
from tests.algorithm_validation_tests.exareme3.conftest import parse_response


def _with_drop_then_winsorizer(payload, winsorizer_strategies):
    payload = deepcopy(payload)
    variables = list(
        dict.fromkeys(
            list(payload["algorithm"].get("x") or [])
            + list(payload["algorithm"].get("y") or [])
        )
    )
    payload["preprocessing"] = [
        {
            "name": "missing_values_handler",
            "parameters": {
                "strategies": {variable: "drop" for variable in variables},
            },
        },
        {
            "name": "outlier_winsorizer",
            "parameters": {
                "strategies": winsorizer_strategies,
            },
        },
    ]
    return payload


OUTLIER_WINSORIZER_ALGORITHM_CASES = [
    pytest.param(
        "describe",
        {
            "inputdata": {
                "data_model": "dementia:0.1",
                "datasets": ["desd-synthdata8"],
                "filters": None,
                "variables": ["lefthippocampus", "rightpallidum"],
            },
            "algorithm": {
                "x": [],
                "y": ["lefthippocampus", "rightpallidum"],
                "parameters": {},
            },
        },
        {"lefthippocampus": "iqr", "rightpallidum": "mad"},
        id="describe",
    ),
    pytest.param(
        "histogram",
        {
            "inputdata": {
                "data_model": "dementia:0.1",
                "datasets": ["desd-synthdata8"],
                "filters": None,
                "variables": ["lefthippocampus"],
            },
            "algorithm": {
                "x": None,
                "y": ["lefthippocampus"],
                "parameters": {
                    "bins": 20,
                },
            },
        },
        {"lefthippocampus": "quantile"},
        id="histogram",
    ),
    pytest.param(
        "describe",
        {
            "inputdata": {
                "data_model": "dementia:0.1",
                "datasets": ["desd-synthdata8"],
                "filters": None,
                "variables": ["lefthippocampus", "rightpallidum", "subjectageyears"],
            },
            "algorithm": {
                "x": [],
                "y": ["lefthippocampus", "rightpallidum", "subjectageyears"],
                "parameters": {},
            },
        },
        {"lefthippocampus": "iqr"},
        id="describe_partial_numeric_winsorizer_subset",
    ),
    pytest.param(
        "pca",
        {
            "inputdata": {
                "data_model": "dementia:0.1",
                "datasets": [
                    "edsd0",
                    "edsd1",
                    "edsd2",
                    "edsd3",
                ],
                "filters": None,
                "variables": ["lefthippocampus", "righthippocampus"],
            },
            "algorithm": {
                "x": None,
                "y": ["lefthippocampus", "righthippocampus"],
                "parameters": None,
            },
        },
        {"lefthippocampus": "iqr", "righthippocampus": "iqr"},
        id="pca",
    ),
    pytest.param(
        "linear_regression",
        {
            "inputdata": {
                "data_model": "dementia:0.1",
                "datasets": [
                    "edsd7",
                    "edsd4",
                    "ppmi2",
                    "desd-synthdata2",
                    "edsd0",
                    "ppmi5",
                    "desd-synthdata8",
                    "ppmi3",
                ],
                "filters": None,
                "variables": [
                    "rightgregyrusrectus",
                    "leftthalamusproper",
                    "rightocpoccipitalpole",
                ],
            },
            "algorithm": {
                "x": ["rightgregyrusrectus", "leftthalamusproper"],
                "y": ["rightocpoccipitalpole"],
                "parameters": {},
            },
        },
        {
            "rightgregyrusrectus": "gaussian",
            "leftthalamusproper": "iqr",
            "rightocpoccipitalpole": "mad",
        },
        id="linear_regression",
    ),
    pytest.param(
        "logistic_regression",
        {
            "inputdata": {
                "data_model": "dementia:0.1",
                "datasets": ["ppmi0", "desd-synthdata0"],
                "filters": None,
                "variables": [
                    "rightttgtransversetemporalgyrus",
                    "leftpinsposteriorinsula",
                    "leftpoparietaloperculum",
                    "rightptplanumtemporale",
                    "leftventraldc",
                    "alzheimerbroadcategory",
                ],
            },
            "algorithm": {
                "x": [
                    "rightttgtransversetemporalgyrus",
                    "leftpinsposteriorinsula",
                    "leftpoparietaloperculum",
                    "rightptplanumtemporale",
                    "leftventraldc",
                ],
                "y": ["alzheimerbroadcategory"],
                "parameters": {
                    "positive_class": "Other",
                },
            },
        },
        {
            "rightttgtransversetemporalgyrus": "iqr",
            "leftpinsposteriorinsula": "iqr",
            "leftpoparietaloperculum": "mad",
            "rightptplanumtemporale": "gaussian",
            "leftventraldc": "quantile",
        },
        id="logistic_regression",
    ),
    pytest.param(
        "kmeans",
        {
            "inputdata": {
                "data_model": "dementia:0.1",
                "datasets": [
                    "edsd0",
                    "edsd1",
                    "edsd2",
                    "edsd3",
                ],
                "filters": None,
                "variables": ["lefthippocampus", "righthippocampus"],
            },
            "algorithm": {
                "x": None,
                "y": ["lefthippocampus", "righthippocampus"],
                "parameters": {
                    "k": 4,
                    "tol": 0.0001,
                    "maxiter": 100,
                },
            },
        },
        {"lefthippocampus": "iqr", "righthippocampus": "iqr"},
        id="kmeans",
    ),
    pytest.param(
        "pearson_correlation",
        {
            "inputdata": {
                "data_model": "dementia:0.1",
                "datasets": [
                    "desd-synthdata8",
                    "edsd8",
                    "ppmi9",
                    "edsd7",
                    "edsd2",
                    "desd-synthdata1",
                ],
                "filters": None,
                "variables": [
                    "rightsplsuperiorparietallobule",
                    "rightttgtransversetemporalgyrus",
                    "leftcaudate",
                ],
            },
            "algorithm": {
                "x": None,
                "y": [
                    "rightsplsuperiorparietallobule",
                    "rightttgtransversetemporalgyrus",
                    "leftcaudate",
                ],
                "parameters": {
                    "alpha": 0.95,
                },
            },
        },
        {
            "rightsplsuperiorparietallobule": "iqr",
            "rightttgtransversetemporalgyrus": "mad",
            "leftcaudate": "gaussian",
        },
        id="pearson_correlation",
    ),
    pytest.param(
        "anova_oneway",
        {
            "inputdata": {
                "data_model": "dementia:0.1",
                "datasets": ["ppmi8"],
                "filters": None,
                "variables": ["ppmicategory", "brainstem"],
            },
            "algorithm": {
                "x": ["ppmicategory"],
                "y": ["brainstem"],
                "parameters": {},
            },
        },
        {"brainstem": "iqr"},
        id="anova_oneway",
    ),
    pytest.param(
        "ttest_independent",
        {
            "inputdata": {
                "data_model": "dementia:0.1",
                "datasets": [
                    "ppmi3",
                    "ppmi2",
                    "desd-synthdata9",
                    "ppmi8",
                    "desd-synthdata5",
                    "desd-synthdata3",
                    "edsd1",
                    "ppmi1",
                    "desd-synthdata2",
                    "desd-synthdata4",
                    "edsd3",
                    "desd-synthdata1",
                    "edsd8",
                    "edsd9",
                    "desd-synthdata7",
                    "ppmi6",
                    "edsd7",
                ],
                "filters": None,
                "variables": ["gender", "lefttmptemporalpole"],
            },
            "algorithm": {
                "x": ["gender"],
                "y": ["lefttmptemporalpole"],
                "parameters": {
                    "alt_hypothesis": "two-sided",
                    "alpha": 0.05,
                    "groupA": "M",
                    "groupB": "F",
                },
            },
        },
        {"lefttmptemporalpole": "iqr"},
        id="ttest_independent",
    ),
    pytest.param(
        "ttest_onesample",
        {
            "inputdata": {
                "data_model": "dementia:0.1",
                "datasets": [
                    "edsd4",
                    "edsd8",
                    "edsd3",
                    "desd-synthdata4",
                    "edsd9",
                ],
                "filters": None,
                "variables": ["leftmcggmiddlecingulategyrus"],
            },
            "algorithm": {
                "x": None,
                "y": ["leftmcggmiddlecingulategyrus"],
                "parameters": {
                    "alt_hypothesis": "greater",
                    "alpha": 0.05,
                    "mu": -1.7510563394418988,
                },
            },
        },
        {"leftmcggmiddlecingulategyrus": "mad"},
        id="ttest_onesample",
    ),
]


@pytest.mark.parametrize(
    "algorithm_name, payload, winsorizer_strategies",
    OUTLIER_WINSORIZER_ALGORITHM_CASES,
)
def test_outlier_winsorizer_runs_after_drop_na_for_many_algorithms(
    algorithm_name,
    payload,
    winsorizer_strategies,
):
    payload = _with_drop_then_winsorizer(payload, winsorizer_strategies)

    response = analysis_request(algorithm_name, payload, drop_na=True)

    assert response.status_code == 200, f"{response.status_code}: {response.content}"
    assert parse_response(response)


def test_outlier_winsorizer_rejects_categorical_variable_after_drop_na():
    payload = _with_drop_then_winsorizer(
        {
            "inputdata": {
                "data_model": "dementia:0.1",
                "datasets": ["desd-synthdata8"],
                "filters": None,
                "variables": ["alzheimerbroadcategory"],
            },
            "algorithm": {
                "x": [],
                "y": ["alzheimerbroadcategory"],
                "parameters": {},
            },
        },
        {"alzheimerbroadcategory": "iqr"},
    )

    response = analysis_request("describe", payload, drop_na=True)

    assert response.status_code == 460, f"{response.status_code}: {response.content}"
    assert b"can only be used for numerical variables" in response.content
