"""Fitted-state validation remains effective without Pydantic models."""

from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from exaflow.algorithms.federated.feature_engineering import MaxAbsScaler
from exaflow.algorithms.federated.feature_engineering import MinMaxScaler
from exaflow.algorithms.federated.feature_engineering import QuantileTransformerApprox
from exaflow.algorithms.federated.feature_engineering import RobustScalerApprox
from exaflow.algorithms.federated.feature_engineering import SimpleImputer
from exaflow.algorithms.federated.feature_engineering import StandardScaler
from exaflow.algorithms.federated.feature_engineering import YeoJohnsonTransformer
from exaflow.algorithms.federated.outliers import OutlierDetector
from exaflow.algorithms.federated.outliers import Winsorizer
from exaflow.algorithms.federated.utils import BadInputError
from tests.standalone_tests.federated_algorithms.utils.dummy_agg_client import (
    DummyAggClient,
)


def _state(factory):
    model = factory()
    model.fit(
        data=pd.DataFrame({"feature": np.linspace(0, 8, 80)}),
        categorical_vars=[],
        numerical_vars=["feature"],
        agg_client=DummyAggClient(),
    )
    return type(model), asdict(model.recipe_)


@pytest.mark.parametrize(
    "factory",
    [
        StandardScaler,
        RobustScalerApprox,
        MinMaxScaler,
        MaxAbsScaler,
        SimpleImputer,
        Winsorizer,
        OutlierDetector,
        YeoJohnsonTransformer,
        lambda: QuantileTransformerApprox(n_quantiles=5, platform_minimum=10),
    ],
)
@pytest.mark.parametrize(
    "mutation", ["missing", "extra", "schema", "categorical", "feature_count"]
)
def test_every_state_owner_rejects_invalid_headers(factory, mutation):
    owner, state = _state(factory)
    if mutation == "missing":
        state.pop("feature_names")
    elif mutation == "extra":
        state["unexpected"] = 1
    elif mutation == "schema":
        state["schema_version"] = "1"
    elif mutation == "categorical":
        state["categorical_features"] = ["not_the_feature"]
    else:
        state["feature_names"] = ["feature", "another"]
    with pytest.raises(BadInputError):
        owner.from_recipe(state)


@pytest.mark.parametrize(
    "factory,key,value",
    [
        (StandardScaler, "scale", [0]),
        (MinMaxScaler, "data_max", [-1]),
        (SimpleImputer, "statistics", [None]),
        (SimpleImputer, "statistics", [np.inf]),
        (Winsorizer, "lower_bounds", [100]),
        (Winsorizer, "upper_quantile", -1),
        (OutlierDetector, "parameters_by_feature", [[0, 0]]),
        (OutlierDetector, "parameters_by_feature", [[np.nan, 1]]),
        (OutlierDetector, "threshold", 0),
        (YeoJohnsonTransformer, "lambdas", [np.nan]),
        (YeoJohnsonTransformer, "max_evaluations", True),
    ],
)
def test_method_specific_state_checks_are_not_lost(factory, key, value):
    owner, state = _state(factory)
    state[key] = value
    with pytest.raises(BadInputError):
        owner.from_recipe(state)


@pytest.mark.parametrize(
    "key,value",
    [
        ("references", [0, 0.5, 0.25, 0.75, 1]),
        ("quantiles", [[0, 2, 1, 3, 4]]),
        ("min_observations_per_interval", 1),
        ("n_bins", 1),
        ("max_refinement_rounds", -1),
    ],
)
def test_quantile_state_rejects_invalid_grid_and_support_policy(key, value):
    owner, state = _state(
        lambda: QuantileTransformerApprox(n_quantiles=5, platform_minimum=10)
    )
    state[key] = value
    with pytest.raises(BadInputError):
        owner.from_recipe(state)


@pytest.mark.parametrize(
    "key,value",
    [
        ("support_counts", [0, 1, 10, 10, 0]),
        ("max_interval_width", -1),
        ("limited_by_minimum_support", "false"),
        ("histogram_rounds", 100),
        ("unexpected", 1),
    ],
)
def test_quantile_diagnostics_are_validated_recursively(key, value):
    owner, state = _state(
        lambda: QuantileTransformerApprox(n_quantiles=5, platform_minimum=10)
    )
    state["fit_diagnostics"][0][key] = value
    with pytest.raises(BadInputError):
        owner.from_recipe(state)


def test_loading_state_revalidates_mutated_dataclass_objects():
    model = StandardScaler()
    model.fit(
        data=pd.DataFrame({"feature": [1.0, 2.0]}),
        categorical_vars=[],
        numerical_vars=["feature"],
        agg_client=DummyAggClient(),
    )
    state = deepcopy(model.recipe_)
    state.scale = [0]
    with pytest.raises(BadInputError):
        StandardScaler.from_recipe(state)


@pytest.mark.parametrize("reason", [[], {}, None])
def test_search_reason_type_is_validated_without_raw_type_errors(reason):
    owner, state = _state(YeoJohnsonTransformer)
    state["fit_diagnostics"][0].update(status="approximate", reason=reason)
    with pytest.raises(BadInputError):
        owner.from_recipe(state)


def test_scaler_replay_rejects_integer_outside_float64_range():
    owner, state = _state(StandardScaler)
    state["scale"] = [10**1000]
    with pytest.raises(BadInputError):
        owner.from_recipe(state)


def test_integer_state_validation_does_not_leak_overflow_error():
    owner, state = _state(YeoJohnsonTransformer)
    state["max_evaluations"] = 10**1000
    with pytest.raises(BadInputError):
        owner.from_recipe(state)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: YeoJohnsonTransformer(lambda_tolerance=10**1000),
        lambda: YeoJohnsonTransformer(likelihood_tolerance=10**1000),
        lambda: OutlierDetector(threshold=10**1000),
        lambda: QuantileTransformerApprox(n_bins=10**1000, platform_minimum=10),
    ],
)
def test_unrepresentable_constructor_parameters_are_bad_input(factory):
    with pytest.raises(BadInputError):
        factory()
