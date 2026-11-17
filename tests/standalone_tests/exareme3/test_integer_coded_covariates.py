"""Algorithms code binary covariates as 0/1 and ordinal covariates as integers.

Runs the algorithms in-process through the real worker path on the test data
models, where ``gender`` is binary (0=M, 1=F), ``agegroup`` is ordinal (1=-50y
... 5=+80y) and ``alzheimerbroadcategory`` is nominal (1=AD, 2=CN, 3=Other, 4=MCI).
Categorical CDEs are stored as integer codes, so feature names carry the codes. The covariate columns are
checked against a pooled statsmodels fit on the same rows.
"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from tests.standalone_tests.lazy_aggregation.test_lazy_aggregation_algorithms import (
    DATASET_FILES,
)
from tests.standalone_tests.lazy_aggregation.test_lazy_aggregation_algorithms import (
    LONGITUDINAL_DATASETS,
)
from tests.standalone_tests.lazy_aggregation.test_lazy_aggregation_algorithms import (
    LONGITUDINAL_PREPROCESSING,
)
from tests.standalone_tests.lazy_aggregation.test_lazy_aggregation_algorithms import (
    _case,
)
from tests.standalone_tests.lazy_aggregation.test_lazy_aggregation_algorithms import (
    _request,
)
from tests.standalone_tests.lazy_aggregation.test_lazy_aggregation_algorithms import (
    _run_algorithm,
)
from tests.standalone_tests.lazy_aggregation.test_lazy_aggregation_algorithms import (  # noqa: F401
    aggregation_server,
)
from tests.standalone_tests.lazy_aggregation.test_lazy_aggregation_algorithms import (  # noqa: F401
    worker_duckdb,
)

TEST_DATA_PATH = Path(__file__).parents[2] / "test_data"
AGE_ORDER = ["1", "2", "3", "4", "5"]
pytestmark = pytest.mark.slow


def _pooled(columns):
    frames = [
        pd.read_csv(TEST_DATA_PATH / "dementia_v_0_1" / f"{dataset}.csv")[columns]
        for dataset in DATASET_FILES
    ]
    return pd.concat(frames).dropna().reset_index(drop=True)


def _run(name, case):
    return _run_algorithm(name, _request(name, case))


def _coefficients(result, names):
    return dict(zip(names, result["coefficients"]))


def _nominal_dummies(series):
    levels = sorted(series.unique())
    return pd.DataFrame({f"{series.name}[{lvl}]": series == lvl for lvl in levels[1:]})


def test_linear_regression_codes_binary_and_ordinal_as_single_columns(
    worker_duckdb,
    aggregation_server,  # noqa: F811
):
    case = _case(
        ["leftamygdala", "gender", "agegroup", "alzheimerbroadcategory"],
        ["lefthippocampus"],
    )
    result = _run("linear_regression", case)

    data = _pooled(
        [
            "leftamygdala",
            "gender",
            "agegroup",
            "alzheimerbroadcategory",
            "lefthippocampus",
        ]
    )
    assert result["indep_vars"] == [
        "Intercept",
        "alzheimerbroadcategory[2]",
        "alzheimerbroadcategory[3]",
        "gender[1]",
        "agegroup",
        "leftamygdala",
    ]
    assert result["n_obs"] == len(data)

    design = pd.concat(
        [
            _nominal_dummies(data.alzheimerbroadcategory),
            (data.gender == 1).rename("gender[1]"),
            (data.agegroup - 1).rename("agegroup"),
            data.leftamygdala,
        ],
        axis=1,
    ).astype(float)
    reference = sm.OLS(data.lefthippocampus, sm.add_constant(design)).fit()
    np.testing.assert_allclose(result["coefficients"], reference.params, rtol=1e-8)
    np.testing.assert_allclose(result["std_err"], reference.bse, rtol=1e-8)

    # The old one-hot coding of gender (reference F, dummy M) is the same model:
    # the M coefficient is the negated F coefficient and the intercept shifts by it.
    old_design = design.drop(columns="gender[1]").assign(
        **{"gender[0]": (data.gender == 0).astype(float)}
    )
    old = sm.OLS(data.lefthippocampus, sm.add_constant(old_design)).fit()
    new = _coefficients(result, result["indep_vars"])
    assert new["gender[1]"] == pytest.approx(-old.params["gender[0]"], rel=1e-8)
    assert new["Intercept"] == pytest.approx(
        old.params["const"] + old.params["gender[0]"], rel=1e-8
    )
    assert new["agegroup"] == pytest.approx(old.params["agegroup"], rel=1e-8)


def test_ordinal_covariate_follows_declared_order(
    worker_duckdb,
    aggregation_server,  # noqa: F811
):
    case = _case(["agegroup"], ["lefthippocampus"])
    result = _run("linear_regression", case)
    data = _pooled(["agegroup", "lefthippocampus"])
    assert result["indep_vars"] == ["Intercept", "agegroup"]

    declared = sm.OLS(
        data.lefthippocampus,
        sm.add_constant((data.agegroup - 1).astype(float)),
    ).fit()
    reversed_order = sm.OLS(
        data.lefthippocampus,
        sm.add_constant((5 - data.agegroup).astype(float)),
    ).fit()
    np.testing.assert_allclose(result["coefficients"], declared.params, rtol=1e-8)
    assert not np.allclose(result["coefficients"], reversed_order.params)


def test_logistic_regression_codes_ordinal_covariate_as_integer(
    worker_duckdb,
    aggregation_server,  # noqa: F811
):
    case = _case(
        ["leftamygdala", "agegroup", "alzheimerbroadcategory"],
        ["gender"],
        {"positive_class": "1"},
    )
    result = _run("logistic_regression", case)
    assert result["indep_vars"] == [
        "Intercept",
        "alzheimerbroadcategory[2]",
        "alzheimerbroadcategory[3]",
        "agegroup",
        "leftamygdala",
    ]

    data = _pooled(["leftamygdala", "agegroup", "alzheimerbroadcategory", "gender"])
    design = pd.concat(
        [
            _nominal_dummies(data.alzheimerbroadcategory),
            (data.agegroup - 1).rename("agegroup"),
            data.leftamygdala,
        ],
        axis=1,
    ).astype(float)
    reference = sm.Logit((data.gender == 1).astype(float), sm.add_constant(design)).fit(
        disp=0
    )
    np.testing.assert_allclose(
        result["summary"]["coefficients"], reference.params, rtol=1e-4, atol=1e-6
    )


def test_binary_covariate_is_one_zero_one_column_in_logistic_regression(
    worker_duckdb,
    aggregation_server,  # noqa: F811
):
    case = _case(
        ["leftamygdala", "alzheimerstatus"], ["gender"], {"positive_class": "1"}
    )
    result = _run("logistic_regression", case)
    assert result["indep_vars"] == [
        "Intercept",
        "alzheimerstatus[1]",
        "leftamygdala",
    ]

    data = _pooled(["leftamygdala", "alzheimerstatus", "gender"])
    design = pd.DataFrame(
        {
            "alzheimerstatus[1]": (data.alzheimerstatus == 1).astype(float),
            "leftamygdala": data.leftamygdala,
        }
    )
    reference = sm.Logit((data.gender == 1).astype(float), sm.add_constant(design)).fit(
        disp=0
    )
    np.testing.assert_allclose(
        result["summary"]["coefficients"], reference.params, rtol=1e-4, atol=1e-6
    )


@pytest.mark.parametrize(
    "name, case, indep_vars",
    [
        (
            "linear_regression_cv",
            _case(
                ["leftamygdala", "gender", "agegroup"],
                ["lefthippocampus"],
                {"n_splits": 3},
            ),
            ["Intercept", "gender[1]", "agegroup", "leftamygdala"],
        ),
        (
            "logistic_regression_cv",
            _case(
                ["leftamygdala", "agegroup", "alzheimerstatus"],
                ["gender"],
                {"positive_class": "1", "n_splits": 3},
            ),
            ["Intercept", "alzheimerstatus[1]", "agegroup", "leftamygdala"],
        ),
        (
            "lmm",
            _case(
                ["lefthippocampus", "gender", "agegroup", "dataset"],
                ["righthippocampus"],
                {"grouping_var": ["dataset"]},
            ),
            ["Intercept", "gender[1]", "agegroup", "lefthippocampus"],
        ),
        (
            "glmm_binary",
            _case(
                ["lefthippocampus", "agegroup", "dataset"],
                ["gender"],
                {"positive_class": "1", "grouping_var": ["dataset"]},
            ),
            ["Intercept", "agegroup", "lefthippocampus"],
        ),
        (
            "glmm_ordinal",
            _case(
                ["lefthippocampus", "gender", "dataset"],
                ["agegroup"],
                {"grouping_var": ["dataset"], "category_order": AGE_ORDER},
            ),
            ["Intercept", "gender[1]", "lefthippocampus"],
        ),
        (
            "cox_regression_classical",
            _case(
                ["alzheimerstatus", "gender", "agegroup", "lefthippocampus"],
                ["timesincebaseline"],
                {"event_var": "alzheimerstatus", "positive_class": "0"},
                datasets=LONGITUDINAL_DATASETS,
                data_model="longitudinal_dementia:0.1",
                preprocessing=[
                    {
                        "name": "longitudinal_transformer",
                        "parameters": {
                            **copy.deepcopy(
                                LONGITUDINAL_PREPROCESSING[0]["parameters"]
                            ),
                            "strategies": {
                                "timesincebaseline": "second",
                                "alzheimerstatus": "second",
                                "lefthippocampus": "first",
                                "gender": "first",
                                "agegroup": "first",
                            },
                        },
                    }
                ],
            ),
            ["gender[1]", "agegroup", "lefthippocampus"],
        ),
        (
            "cox_regression_stacked",
            _case(
                ["alzheimerstatus", "gender", "agegroup", "lefthippocampus"],
                ["timesincebaseline"],
                {
                    "event_var": "alzheimerstatus",
                    "positive_class": "0",
                    "time_grid_strategy": "distinct_event_times",
                    "n_time_bins": 10,
                },
                datasets=LONGITUDINAL_DATASETS,
                data_model="longitudinal_dementia:0.1",
                preprocessing=[
                    {
                        "name": "longitudinal_transformer",
                        "parameters": {
                            **copy.deepcopy(
                                LONGITUDINAL_PREPROCESSING[0]["parameters"]
                            ),
                            "strategies": {
                                "timesincebaseline": "second",
                                "alzheimerstatus": "second",
                                "lefthippocampus": "first",
                                "gender": "first",
                                "agegroup": "first",
                            },
                        },
                    }
                ],
            ),
            ["gender[1]", "agegroup", "lefthippocampus"],
        ),
    ],
)
def test_binary_and_ordinal_covariates_are_single_columns(
    name,
    case,
    indep_vars,
    worker_duckdb,
    aggregation_server,  # noqa: F811
):
    result = _run(name, case)
    assert result["indep_vars"] == indep_vars
