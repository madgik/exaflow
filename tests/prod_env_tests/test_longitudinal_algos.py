from copy import deepcopy

from tests.algorithm_validation_tests.exareme3.conftest import algorithm_request

base_input = {
    "inputdata": {
        "y": None,
        "x": None,
        "data_model": "longitudinal_dementia:0.1",
        "datasets": [
            "longitudinal_dementia0",
            "longitudinal_dementia1",
            "longitudinal_dementia2",
        ],
        "filters": None,
    },
    "parameters": {},
    "preprocessing": {
        "longitudinal_transformer": {
            "visit1": "BL",
            "visit2": "FL1",
            "strategies": None,
        },
    },
}


def test_longitudinal_anova_oneway():
    input = deepcopy(base_input)
    input["inputdata"]["x"] = ["gender"]
    input["inputdata"]["y"] = ["lefthippocampus"]
    input["preprocessing"]["longitudinal_transformer"]["strategies"] = {
        "gender": "first",
        "lefthippocampus": "diff",
    }
    response = algorithm_request("anova_oneway", input)
    assert response.status_code == 200, f"{response.status_code}: {response.content}"


def test_longitudinal_anova_twoway():
    input = deepcopy(base_input)
    input["inputdata"]["x"] = ["gender", "agegroup"]
    input["inputdata"]["y"] = ["lefthippocampus"]
    input["parameters"]["sstype"] = 2
    input["preprocessing"]["longitudinal_transformer"]["strategies"] = {
        "gender": "first",
        "lefthippocampus": "diff",
        "agegroup": "first",
    }
    response = algorithm_request("anova_twoway", input)
    assert response.status_code == 200, f"{response.status_code}: {response.content}"


def test_longitudinal_linear_regression():
    input = deepcopy(base_input)
    input["inputdata"]["x"] = ["lefthippocampus"]
    input["inputdata"]["y"] = ["righthippocampus"]
    input["preprocessing"]["longitudinal_transformer"]["strategies"] = {
        "righthippocampus": "first",
        "lefthippocampus": "diff",
    }
    response = algorithm_request("linear_regression", input)
    assert response.status_code == 200, f"{response.status_code}: {response.content}"


def test_longitudinal_linear_regression_cv():
    input = deepcopy(base_input)
    input["inputdata"]["x"] = ["lefthippocampus"]
    input["inputdata"]["y"] = ["righthippocampus"]
    input["parameters"]["n_splits"] = 2
    input["preprocessing"]["longitudinal_transformer"]["strategies"] = {
        "righthippocampus": "first",
        "lefthippocampus": "diff",
    }
    response = algorithm_request("linear_regression_cv", input)
    assert response.status_code == 200, f"{response.status_code}: {response.content}"


def test_longitudinal_logistic_regression():
    input = deepcopy(base_input)
    input["inputdata"]["x"] = ["lefthippocampus", "leftamygdala"]
    input["inputdata"]["y"] = ["gender"]
    input["parameters"]["positive_class"] = "F"
    input["preprocessing"]["longitudinal_transformer"]["strategies"] = {
        "gender": "second",
        "leftamygdala": "first",
        "lefthippocampus": "diff",
    }
    response = algorithm_request("logistic_regression", input)
    assert response.status_code == 200, f"{response.status_code}: {response.content}"


def test_longitudinal_logistic_regression_error_less_strategies():
    input = deepcopy(base_input)
    input["inputdata"]["x"] = [
        "cerebellarvermallobulesviiix",
        "rightpcggposteriorcingulategyrus",
        "leftacgganteriorcingulategyrus",
    ]
    input["inputdata"]["y"] = ["alzheimerbroadcategory"]
    input["parameters"]["positive_class"] = "Other"
    input["preprocessing"]["longitudinal_transformer"]["strategies"] = {
        "cerebellarvermallobulesviiix": "first",
        "rightpcggposteriorcingulategyrus": "diff",
        "leftacgganteriorcingulategyrus": "diff",
    }

    response = algorithm_request("logistic_regression", input)
    assert response.status_code == 460, f"{response.status_code}: {response.content}"


def test_longitudinal_logistic_regression_cv():
    input = deepcopy(base_input)
    input["inputdata"]["x"] = ["lefthippocampus", "leftamygdala"]
    input["inputdata"]["y"] = ["gender"]
    input["parameters"]["n_splits"] = 2
    input["parameters"]["positive_class"] = "F"
    input["preprocessing"]["longitudinal_transformer"]["strategies"] = {
        "gender": "second",
        "leftamygdala": "first",
        "lefthippocampus": "diff",
    }
    response = algorithm_request("logistic_regression_cv", input)
    assert response.status_code == 200, f"{response.status_code}: {response.content}"


def test_longitudinal_naive_bayes_gaussian_cv():
    input = deepcopy(base_input)
    input["inputdata"]["x"] = ["lefthippocampus", "leftamygdala"]
    input["inputdata"]["y"] = ["gender"]
    input["parameters"]["n_splits"] = 2
    input["preprocessing"]["longitudinal_transformer"]["strategies"] = {
        "gender": "second",
        "leftamygdala": "first",
        "lefthippocampus": "diff",
    }
    response = algorithm_request("naive_bayes_gaussian_cv", input)
    assert response.status_code == 200, f"{response.status_code}: {response.content}"


def test_longitudinal_naive_bayes_categorical_cv():
    input = deepcopy(base_input)
    input["inputdata"]["x"] = ["agegroup"]
    input["inputdata"]["y"] = ["gender"]
    input["parameters"]["n_splits"] = 2
    input["preprocessing"]["longitudinal_transformer"]["strategies"] = {
        "gender": "second",
        "agegroup": "first",
    }
    response = algorithm_request("naive_bayes_categorical_cv", input)
    assert response.status_code == 200, f"{response.status_code}: {response.content}"
