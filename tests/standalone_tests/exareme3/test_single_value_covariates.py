"""A binary or ordinal covariate with one observed value fails every algorithm.

The covariate is made constant by filtering the data model, as in a user request.
"""

import copy

import pytest

from exaflow.worker_communication import BadUserInput
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

pytestmark = pytest.mark.slow

AGE_ORDER = ["1", "2", "3", "4", "5"]


def _equal(variable, value):
    return {
        "condition": "AND",
        "rules": [
            {
                "id": variable,
                "field": variable,
                "type": "integer",
                "input": "number",
                "operator": "equal",
                "value": value,
            }
        ],
    }


def _filtered(case, variable, value):
    return {**case, "filters": _equal(variable, value)}


def _cox_case(**parameters):
    strategies = {
        "timesincebaseline": "second",
        "alzheimerstatus": "second",
        "lefthippocampus": "first",
        "gender": "first",
        "agegroup": "first",
    }
    return _case(
        ["alzheimerstatus", "gender", "agegroup", "lefthippocampus"],
        ["timesincebaseline"],
        {"event_var": "alzheimerstatus", "positive_class": "0", **parameters},
        datasets=LONGITUDINAL_DATASETS,
        data_model="longitudinal_dementia:0.1",
        preprocessing=[
            {
                "name": "longitudinal_transformer",
                "parameters": {
                    **copy.deepcopy(LONGITUDINAL_PREPROCESSING[0]["parameters"]),
                    "strategies": strategies,
                },
            }
        ],
    )


CONSTANT_BINARY = ("gender", 1)  # every row is F
CONSTANT_ORDINAL = ("agegroup", 3)

CASES = [
    pytest.param(
        "linear_regression",
        _case(["leftamygdala", "gender"], ["lefthippocampus"]),
        CONSTANT_BINARY,
        id="linear-binary",
    ),
    pytest.param(
        "linear_regression",
        _case(["leftamygdala", "agegroup"], ["lefthippocampus"]),
        CONSTANT_ORDINAL,
        id="linear-ordinal",
    ),
    pytest.param(
        "linear_regression_cv",
        _case(["leftamygdala", "gender"], ["lefthippocampus"], {"n_splits": 3}),
        CONSTANT_BINARY,
        id="linear_cv-binary",
    ),
    pytest.param(
        "logistic_regression",
        _case(["leftamygdala", "alzheimerstatus"], ["gender"], {"positive_class": "1"}),
        ("alzheimerstatus", 0),
        id="logistic-binary",
    ),
    pytest.param(
        "logistic_regression",
        _case(["leftamygdala", "agegroup"], ["gender"], {"positive_class": "1"}),
        CONSTANT_ORDINAL,
        id="logistic-ordinal",
    ),
    pytest.param(
        "logistic_regression_cv",
        _case(
            ["leftamygdala", "agegroup"],
            ["gender"],
            {"positive_class": "1", "n_splits": 3},
        ),
        CONSTANT_ORDINAL,
        id="logistic_cv-ordinal",
    ),
    pytest.param(
        "lmm",
        _case(
            ["lefthippocampus", "gender", "dataset"],
            ["righthippocampus"],
            {"grouping_var": ["dataset"]},
        ),
        CONSTANT_BINARY,
        id="lmm-binary",
    ),
    pytest.param(
        "glmm_binary",
        _case(
            ["lefthippocampus", "agegroup", "dataset"],
            ["gender"],
            {"positive_class": "1", "grouping_var": ["dataset"]},
        ),
        CONSTANT_ORDINAL,
        id="glmm_binary-ordinal",
    ),
    pytest.param(
        "glmm_ordinal",
        _case(
            ["lefthippocampus", "gender", "dataset"],
            ["agegroup"],
            {"grouping_var": ["dataset"], "category_order": AGE_ORDER},
        ),
        CONSTANT_BINARY,
        id="glmm_ordinal-binary",
    ),
    pytest.param(
        "cox_regression_classical", _cox_case(), CONSTANT_BINARY, id="cox-binary"
    ),
    pytest.param(
        "cox_regression_stacked",
        _cox_case(time_grid_strategy="distinct_event_times", n_time_bins=10),
        CONSTANT_ORDINAL,
        id="cox_stacked-ordinal",
    ),
]


@pytest.mark.parametrize("name, case, constant", CASES)
def test_constant_covariate_stops_the_analysis(
    name,
    case,
    constant,
    worker_duckdb,
    aggregation_server,  # noqa: F811
):
    variable, value = constant
    request = _request(name, _filtered(case, variable, value))

    with pytest.raises(BadUserInput) as error:
        _run_algorithm(name, request)

    assert f"Covariate '{variable}' has only one observed value" in str(error.value)
    assert "Remove it or change the filters." in str(error.value)


def test_all_female_reproduction_reports_the_single_value(
    worker_duckdb,
    aggregation_server,  # noqa: F811
):
    case = _filtered(
        _case(["leftamygdala", "gender"], ["lefthippocampus"]), "gender", 1
    )

    with pytest.raises(BadUserInput) as error:
        _run_algorithm("linear_regression", _request("linear_regression", case))

    assert str(error.value) == (
        "Covariate 'gender' has only one observed value (1) after filtering; "
        "it cannot be estimated. Remove it or change the filters."
    )
