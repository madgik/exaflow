"""Covariates are coded by CDE type: binary 0/1, ordinal integers, nominal one-hot."""

import numpy as np
import pandas as pd
import pytest

from exaflow.algorithms.exareme3.utils.covariate_encoding import (
    build_covariate_transformer,
)
from exaflow.algorithms.federated.utils import BadInputError
from exaflow.datatypes import CDEType


class _UnionClient:
    def union(self, values):
        return sorted(set(values))


def _cde(cde_type, levels):
    return {"type": cde_type, "enumerations": {level: level for level in levels}}


METADATA = {
    # Declared order deliberately differs from alphabetical order.
    "sex": _cde("binary", ["M", "F"]),
    "grade": _cde("ordinal", ["low", "mid", "high"]),
    "site": _cde("nominal", ["s2", "s1", "s3"]),
    "pair": _cde("nominal", ["x", "y"]),
    "age": {"type": "continuous"},
    "odd_binary": _cde("binary", ["a", "b", "c"]),
    "lonely_ordinal": _cde("ordinal", ["only"]),
}


def _data():
    return pd.DataFrame(
        {
            "sex": ["F", "M", "F", "M"],
            "grade": ["high", "low", "mid", "low"],
            "site": ["s1", "s2", "s3", "s1"],
            "pair": ["y", "x", "x", "y"],
            "age": [50.0, 60.0, 70.0, 80.0],
        }
    )


def _encode(categorical_vars, numerical_vars=(), data=None):
    data = _data() if data is None else data
    transformer = build_covariate_transformer(categorical_vars, METADATA)
    transformer.fit(
        agg_client=_UnionClient(),
        data=data,
        categorical_vars=list(categorical_vars),
        numerical_vars=list(numerical_vars),
    )
    X = transformer.transform(
        data,
        categorical_vars=list(categorical_vars),
        numerical_vars=list(numerical_vars),
    )
    names = transformer.get_feature_names_out(
        categorical_vars=list(categorical_vars),
        numerical_vars=list(numerical_vars),
    )
    return X, names


def test_binary_is_a_single_zero_one_column_in_declared_order():
    X, names = _encode(["sex"])
    assert names == ["sex[F]"]
    assert X.shape == (4, 1)
    # Declared order is M, F, so M is 0 and F is 1 (alphabetical order would flip it).
    np.testing.assert_array_equal(X[:, 0], [1, 0, 1, 0])


def test_ordinal_is_one_integer_column_in_declared_order():
    X, names = _encode(["grade"])
    assert names == ["grade"]
    assert X.shape == (4, 1)
    # Declared order is low, mid, high (alphabetical order would be high, low, mid).
    np.testing.assert_array_equal(X[:, 0], [2, 0, 1, 0])


def test_nominal_is_still_one_hot_with_first_sorted_level_as_reference():
    X, names = _encode(["site"])
    assert names == ["site[s2]", "site[s3]"]
    np.testing.assert_array_equal(X, [[0, 0], [1, 0], [0, 1], [0, 0]])


def test_two_level_nominal_is_one_hot_not_binary_coded():
    X, names = _encode(["pair"])
    assert names == ["pair[y]"]
    np.testing.assert_array_equal(X[:, 0], [1, 0, 0, 1])


def test_mixed_covariates_layout_is_nominal_binary_ordinal_numerical():
    X, names = _encode(["grade", "sex", "site"], ["age"])
    assert names == ["site[s2]", "site[s3]", "sex[F]", "grade", "age"]
    assert X.shape == (4, 5)
    np.testing.assert_array_equal(X[:, 2], [1, 0, 1, 0])
    np.testing.assert_array_equal(X[:, 3], [2, 0, 1, 0])
    np.testing.assert_array_equal(X[:, 4], [50, 60, 70, 80])


def test_only_numerical_covariates_pass_through():
    X, names = _encode([], ["age"])
    assert names == ["age"]
    np.testing.assert_array_equal(X[:, 0], [50, 60, 70, 80])


def test_undeclared_level_raises():
    data = _data()
    data.loc[0, "grade"] = "extreme"
    with pytest.raises(BadInputError, match="Unknown categories.*'grade'"):
        _encode(["grade"], data=data)
    data = _data()
    data.loc[1, "sex"] = "X"
    with pytest.raises(BadInputError, match="Unknown categories.*'sex'"):
        _encode(["sex"], data=data)


def test_missing_value_raises_instead_of_being_coded():
    data = _data()
    data.loc[2, "sex"] = None
    with pytest.raises(BadInputError, match="Unknown categories.*'sex'"):
        _encode(["sex"], data=data)


def test_binary_must_declare_two_levels():
    with pytest.raises(BadInputError, match="exactly two levels"):
        build_covariate_transformer(["odd_binary"], METADATA)


def test_ordinal_must_declare_at_least_two_levels():
    with pytest.raises(BadInputError, match="at least two levels"):
        build_covariate_transformer(["lonely_ordinal"], METADATA)


def test_non_categorical_variable_is_rejected():
    with pytest.raises(BadInputError, match="not categorical"):
        build_covariate_transformer(["age"], METADATA)


def test_non_categorical_error_shows_the_plain_type_value():
    with pytest.raises(BadInputError, match="has type 'continuous'"):
        build_covariate_transformer(["age"], {"age": {"type": CDEType.CONTINUOUS}})


def test_reference_design_matrix_used_to_generate_fixtures_matches_the_algorithms():
    from tests.testcase_generators.design_matrix import coded_design_matrix

    data = _data()
    cdes = [
        {
            "code": code,
            "type": cde["type"],
            "enumerations": [{"code": level} for level in cde.get("enumerations", [])],
        }
        for code, cde in METADATA.items()
    ]
    categorical = ["grade", "sex", "site", "pair"]
    X, names = _encode(categorical, ["age"])
    reference = coded_design_matrix(data[categorical + ["age"]], cdes)
    assert list(reference.columns) == names
    np.testing.assert_array_equal(reference.to_numpy(), X)


@pytest.mark.parametrize("variable, value", [("sex", "F"), ("grade", "mid")])
def test_covariate_with_a_single_observed_value_stops_the_analysis(variable, value):
    data = _data()
    data[variable] = value
    with pytest.raises(BadInputError) as error:
        _encode([variable], data=data)
    assert str(error.value) == (
        f"Covariate '{variable}' has only one observed value ({value}) after "
        "filtering; it cannot be estimated. Remove it or change the filters."
    )


def test_observed_variation_check_can_be_disabled_for_cv_folds():
    data = _data()
    data["sex"] = "F"
    with pytest.raises(BadInputError, match="only one observed value"):
        build_covariate_transformer(["sex"], METADATA).fit(
            agg_client=_UnionClient(),
            data=data,
            categorical_vars=["sex"],
            numerical_vars=[],
        )

    transformer = build_covariate_transformer(
        ["sex"], METADATA, check_observed_variation=False
    )
    transformer.fit(
        agg_client=_UnionClient(),
        data=data,
        categorical_vars=["sex"],
        numerical_vars=[],
    )
    X = transformer.transform(data, categorical_vars=["sex"], numerical_vars=[])
    np.testing.assert_array_equal(X[:, 0], [1, 1, 1, 1])


def test_covariate_without_observed_values_stops_the_analysis():
    data = _data()
    data["sex"] = None
    with pytest.raises(BadInputError, match="'sex' has no observed values"):
        _encode(["sex"], data=data)


class _TwoWorkerClient:
    """Pretends another worker observed the level this worker does not see."""

    def __init__(self, other_values):
        self._other = other_values

    def union(self, values):
        return sorted(set(values) | set(self._other))


def test_observed_values_are_united_across_workers():
    data = _data()
    data["sex"] = "F"
    transformer = build_covariate_transformer(["sex"], METADATA)
    transformer.fit(
        agg_client=_TwoWorkerClient(["M"]),
        data=data,
        categorical_vars=["sex"],
        numerical_vars=[],
    )
    assert transformer.get_feature_names_out(
        categorical_vars=["sex"], numerical_vars=[]
    ) == ["sex[F]"]


def test_nominal_covariate_with_a_single_value_is_not_rejected():
    data = _data()
    data["site"] = "s1"
    X, names = _encode(["site"], data=data)
    assert names == []
