"""Binary and ordinal slots match CDE types one-to-one; nominal is never converted."""

from unittest.mock import Mock

import pytest

from exaflow import exareme3_algorithm_classes
from exaflow.algorithms.exareme3.mixed_effects.glmm_ordinal import GLMMOrdinal
from exaflow.algorithms.exareme3.utils.cde_types import declared_levels
from exaflow.algorithms.exareme3.utils.cde_types import require_cde_type
from exaflow.algorithms.exareme3.utils.cde_types import require_declared_level
from exaflow.algorithms.specifications import InputDataStatType
from exaflow.datatypes import CDEType
from exaflow.datatypes import is_integer_coded
from exaflow.worker_communication import BadUserInput

BINARY = [InputDataStatType.BINARY]
ORDINAL = [InputDataStatType.ORDINAL]


def _cde(cde_type, levels):
    return {
        "type": cde_type,
        "enumerations": {level: level for level in levels},
    }


def _metadata(**overrides):
    metadata = {
        "outcome": _cde("binary", ["no", "yes"]),
        "nominal2": _cde("nominal", ["a", "b"]),
        "nominal3": _cde("nominal", ["a", "b", "c"]),
        "grade": _cde("ordinal", ["low", "mid", "high"]),
        "bad_binary": _cde("binary", ["a", "b", "c"]),
        "age": {"type": "continuous"},
        "site": _cde("nominal", ["s1", "s2"]),
    }
    metadata.update(overrides)
    return metadata


def _algorithm(name, *, x, y, parameters, metadata=None):
    engine = Mock()
    algorithm = exareme3_algorithm_classes[name](
        engine=engine,
        logger=Mock(),
        inputdata=Mock(),
        x=x,
        y=y,
        metadata=metadata or _metadata(),
        parameters=parameters,
    )
    return algorithm, engine


@pytest.mark.parametrize(
    "name",
    ["logistic_regression", "logistic_regression_cv", "glmm_binary"],
)
def test_binary_outcome_algorithms_only_accept_binary_outcomes(name):
    spec = exareme3_algorithm_classes[name].get_specification()
    assert spec.y.types == BINARY


def test_fisher_exact_only_accepts_binary_variables():
    spec = exareme3_algorithm_classes["fisher_exact"].get_specification()
    assert spec.x.types == BINARY
    assert spec.y.types == BINARY


def test_glmm_ordinal_only_accepts_ordinal_outcome():
    spec = GLMMOrdinal.get_specification()
    assert spec.y.types == ORDINAL
    assert spec.parameters["category_order"].required is False


@pytest.mark.parametrize("name", ["cox_regression_classical", "cox_regression_stacked"])
def test_cox_event_variable_must_be_binary(name):
    spec = exareme3_algorithm_classes[name].get_specification()
    assert spec.parameters["event_var"].variable_types == BINARY


@pytest.mark.parametrize(
    "name, x, parameters",
    [
        ("logistic_regression", ["age"], {"positive_class": "a"}),
        ("logistic_regression_cv", ["age"], {"positive_class": "a", "n_splits": 3}),
        (
            "glmm_binary",
            ["age", "site"],
            {"positive_class": "a", "grouping_var": ["site"]},
        ),
    ],
)
@pytest.mark.parametrize("outcome", ["nominal2", "nominal3", "grade", "age"])
def test_binary_outcome_algorithms_reject_other_types_at_runtime(
    name, x, parameters, outcome
):
    algorithm, engine = _algorithm(name, x=x, y=[outcome], parameters=parameters)

    with pytest.raises(BadUserInput, match="requires a 'binary' variable"):
        algorithm.run()

    engine.run_udf.assert_not_called()


def test_binary_outcome_with_wrong_number_of_levels_is_rejected():
    algorithm, engine = _algorithm(
        "logistic_regression",
        x=["age"],
        y=["bad_binary"],
        parameters={"positive_class": "a"},
    )

    with pytest.raises(BadUserInput, match="exactly two"):
        algorithm.run()

    engine.run_udf.assert_not_called()


@pytest.mark.parametrize("name", ["logistic_regression", "glmm_binary"])
def test_positive_class_must_be_a_declared_level_of_the_binary_outcome(name):
    parameters = {"positive_class": "maybe", "grouping_var": ["site"]}
    algorithm, engine = _algorithm(
        name, x=["age", "site"], y=["outcome"], parameters=parameters
    )

    with pytest.raises(BadUserInput, match="positive_class"):
        algorithm.run()

    engine.run_udf.assert_not_called()


@pytest.mark.parametrize("name", ["cox_regression_classical", "cox_regression_stacked"])
def test_cox_positive_class_is_required_like_logistic_regression(name):
    cox = exareme3_algorithm_classes[name].get_specification().parameters
    logistic = (
        exareme3_algorithm_classes["logistic_regression"].get_specification().parameters
    )
    assert cox["positive_class"].required is True
    assert cox["positive_class"].required == logistic["positive_class"].required
    assert cox["positive_class"].types == logistic["positive_class"].types


@pytest.mark.parametrize("name", ["cox_regression_classical", "cox_regression_stacked"])
@pytest.mark.parametrize("positive_class", ["maybe", None])
def test_cox_positive_class_must_be_a_declared_level_of_the_event_variable(
    name, positive_class
):
    algorithm, engine = _algorithm(
        name,
        x=["outcome", "age"],
        y=["age"],
        parameters={"event_var": "outcome", "positive_class": positive_class},
    )

    with pytest.raises(BadUserInput, match="positive_class"):
        algorithm.run()

    engine.run_udf.assert_not_called()


@pytest.mark.parametrize("name", ["cox_regression_classical", "cox_regression_stacked"])
@pytest.mark.parametrize("event_var", ["nominal2", "nominal3", "grade"])
def test_cox_rejects_non_binary_event_variable_at_runtime(name, event_var):
    algorithm, engine = _algorithm(
        name,
        x=[event_var, "age"],
        y=["age"],
        parameters={"event_var": event_var, "positive_class": "a"},
    )

    with pytest.raises(BadUserInput, match="event variable"):
        algorithm.run()

    engine.run_udf.assert_not_called()


@pytest.mark.parametrize(
    "x_var, y_var", [("nominal2", "outcome"), ("outcome", "nominal2")]
)
def test_fisher_exact_rejects_nominal_variables_at_runtime(x_var, y_var):
    algorithm, engine = _algorithm("fisher_exact", x=[x_var], y=[y_var], parameters={})

    with pytest.raises(BadUserInput, match="requires a 'binary' variable"):
        algorithm.run()

    engine.run_udf.assert_not_called()


@pytest.mark.parametrize("outcome", ["nominal2", "nominal3", "outcome", "age"])
def test_glmm_ordinal_rejects_non_ordinal_outcome_at_runtime(outcome):
    algorithm, engine = _algorithm(
        "glmm_ordinal",
        x=["age", "site"],
        y=[outcome],
        parameters={
            "grouping_var": ["site"],
            "category_order": ["a", "b", "c"],
        },
    )

    with pytest.raises(BadUserInput, match="requires a 'ordinal' variable"):
        algorithm.run()

    engine.run_udf.assert_not_called()


def _glmm_ordinal(parameters):
    algorithm, _ = _algorithm(
        "glmm_ordinal", x=["age", "site"], y=["grade"], parameters=parameters
    )
    return algorithm


def test_glmm_ordinal_defaults_category_order_to_the_declared_order():
    assert _glmm_ordinal({})._resolve_category_order("grade") == [
        "low",
        "mid",
        "high",
    ]


def test_glmm_ordinal_accepts_category_order_equal_to_the_declared_order():
    algorithm = _glmm_ordinal({"category_order": ["low", "mid", "high"]})

    assert algorithm._resolve_category_order("grade") == ["low", "mid", "high"]


@pytest.mark.parametrize(
    "category_order",
    [["high", "mid", "low"], ["low", "high"], ["low", "mid", "high", "extra"]],
)
def test_glmm_ordinal_rejects_category_order_that_differs_from_the_declared_order(
    category_order,
):
    algorithm = _glmm_ordinal({"category_order": category_order})

    with pytest.raises(BadUserInput, match="cannot be overridden"):
        algorithm._resolve_category_order("grade")


def test_require_cde_type_accepts_exact_type_only():
    metadata = _metadata()

    require_cde_type(metadata, "outcome", CDEType.BINARY, role="outcome")
    require_cde_type(metadata, "grade", CDEType.ORDINAL, role="outcome")
    for variable in ("nominal2", "grade", "age"):
        with pytest.raises(BadUserInput, match=f"'{variable}'"):
            require_cde_type(metadata, variable, CDEType.BINARY, role="outcome")
    with pytest.raises(BadUserInput, match="'outcome'"):
        require_cde_type(metadata, "outcome", CDEType.ORDINAL, role="outcome")
    with pytest.raises(BadUserInput, match="'missing'"):
        require_cde_type(metadata, "missing", CDEType.BINARY, role="outcome")


def test_declared_levels_follow_the_declared_order():
    assert declared_levels(_metadata(), "grade") == ["low", "mid", "high"]


def test_require_declared_level_compares_as_text():
    metadata = _metadata(flag=_cde("binary", ["0", "1"]))

    require_declared_level(metadata, "flag", 1, parameter="positive_class")
    with pytest.raises(BadUserInput, match="positive_class"):
        require_declared_level(metadata, "flag", 2, parameter="positive_class")


def test_integer_codes_must_be_canonical():
    assert not is_integer_coded("nominal", ["01", "02"])
    assert is_integer_coded("nominal", ["1", "2"])
    assert not is_integer_coded("nominal", [" 2 ", "+1", "-0"])
