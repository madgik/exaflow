from copy import deepcopy

from tests.algorithm_validation_tests.exareme3.conftest import algorithm_request

base_input = {
    "inputdata": {
        "data_model": "longitudinal_dementia:0.1",
        "datasets": [
            "longitudinal_dementia0",
            "longitudinal_dementia1",
            "longitudinal_dementia2",
        ],
        "filters": None,
        "variables": [],
    },
    "preprocessing": [
        {
            "name": "longitudinal_transformer",
            "parameters": {
                "visit1": "BL",
                "visit2": "FL1",
                "strategies": None,
            },
        }
    ],
    "algorithm": {"x": None, "y": None, "parameters": {}},
}


def _set_algorithm_inputs(input_, *, x, y):
    input_["algorithm"]["x"] = x
    input_["algorithm"]["y"] = y
    input_["inputdata"]["variables"] = list(dict.fromkeys(x + y))


def _set_parameters(input_, **parameters):
    input_["algorithm"]["parameters"].update(parameters)


def _set_longitudinal_strategies(input_, strategies):
    input_["preprocessing"][0]["parameters"]["strategies"] = strategies


def test_longitudinal_anova_oneway():
    input = deepcopy(base_input)
    _set_algorithm_inputs(input, x=["gender"], y=["lefthippocampus"])
    _set_longitudinal_strategies(
        input,
        {
            "gender": "first",
            "lefthippocampus": "diff",
        },
    )
    response = algorithm_request("anova_oneway", input)
    assert response.status_code == 200, f"{response.status_code}: {response.content}"


def test_longitudinal_anova_twoway():
    input = deepcopy(base_input)
    _set_algorithm_inputs(input, x=["gender", "agegroup"], y=["lefthippocampus"])
    _set_parameters(input, sstype=2)
    _set_longitudinal_strategies(
        input,
        {
            "gender": "first",
            "lefthippocampus": "diff",
            "agegroup": "first",
        },
    )
    response = algorithm_request("anova_twoway", input)
    assert response.status_code == 200, f"{response.status_code}: {response.content}"


def test_longitudinal_linear_regression():
    input = deepcopy(base_input)
    _set_algorithm_inputs(input, x=["lefthippocampus"], y=["righthippocampus"])
    _set_longitudinal_strategies(
        input,
        {
            "righthippocampus": "first",
            "lefthippocampus": "diff",
        },
    )
    response = algorithm_request("linear_regression", input)
    assert response.status_code == 200, f"{response.status_code}: {response.content}"


def test_longitudinal_linear_regression_cv():
    input = deepcopy(base_input)
    _set_algorithm_inputs(input, x=["lefthippocampus"], y=["righthippocampus"])
    _set_parameters(input, n_splits=2)
    _set_longitudinal_strategies(
        input,
        {
            "righthippocampus": "first",
            "lefthippocampus": "diff",
        },
    )
    response = algorithm_request("linear_regression_cv", input)
    assert response.status_code == 200, f"{response.status_code}: {response.content}"


def test_longitudinal_logistic_regression():
    input = deepcopy(base_input)
    _set_algorithm_inputs(input, x=["lefthippocampus", "leftamygdala"], y=["gender"])
    _set_parameters(input, positive_class="F")
    _set_longitudinal_strategies(
        input,
        {
            "gender": "second",
            "leftamygdala": "first",
            "lefthippocampus": "diff",
        },
    )
    response = algorithm_request("logistic_regression", input)
    assert response.status_code == 200, f"{response.status_code}: {response.content}"


def test_longitudinal_logistic_regression_error_less_strategies():
    input = deepcopy(base_input)
    _set_algorithm_inputs(
        input,
        x=[
            "cerebellarvermallobulesviiix",
            "rightpcggposteriorcingulategyrus",
            "leftacgganteriorcingulategyrus",
        ],
        y=["alzheimerbroadcategory"],
    )
    _set_parameters(input, positive_class="Other")
    _set_longitudinal_strategies(
        input,
        {
            "cerebellarvermallobulesviiix": "first",
            "rightpcggposteriorcingulategyrus": "diff",
            "leftacgganteriorcingulategyrus": "diff",
        },
    )

    response = algorithm_request("logistic_regression", input)
    assert response.status_code == 460, f"{response.status_code}: {response.content}"


def test_longitudinal_logistic_regression_cv():
    input = deepcopy(base_input)
    _set_algorithm_inputs(input, x=["lefthippocampus", "leftamygdala"], y=["gender"])
    _set_parameters(input, n_splits=2, positive_class="F")
    _set_longitudinal_strategies(
        input,
        {
            "gender": "second",
            "leftamygdala": "first",
            "lefthippocampus": "diff",
        },
    )
    response = algorithm_request("logistic_regression_cv", input)
    assert response.status_code == 200, f"{response.status_code}: {response.content}"


def test_longitudinal_naive_bayes_gaussian_cv():
    input = deepcopy(base_input)
    _set_algorithm_inputs(input, x=["lefthippocampus", "leftamygdala"], y=["gender"])
    _set_parameters(input, n_splits=2)
    _set_longitudinal_strategies(
        input,
        {
            "gender": "second",
            "leftamygdala": "first",
            "lefthippocampus": "diff",
        },
    )
    response = algorithm_request("naive_bayes_gaussian_cv", input)
    assert response.status_code == 200, f"{response.status_code}: {response.content}"


def test_longitudinal_naive_bayes_categorical_cv():
    input = deepcopy(base_input)
    _set_algorithm_inputs(input, x=["agegroup"], y=["gender"])
    _set_parameters(input, n_splits=2)
    _set_longitudinal_strategies(
        input,
        {
            "gender": "second",
            "agegroup": "first",
        },
    )
    response = algorithm_request("naive_bayes_categorical_cv", input)
    assert response.status_code == 200, f"{response.status_code}: {response.content}"
