"""Exercise the actual public DataFrame/Protocol contract, without adapters."""

import inspect
import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from exaflow.algorithms.federated import FederatedColumnTransformer
from exaflow.algorithms.federated import FederatedPipeline
from exaflow.algorithms.federated.feature_engineering import AsinhTransformer
from exaflow.algorithms.federated.feature_engineering import Log1pTransformer
from exaflow.algorithms.federated.feature_engineering import MaxAbsScaler
from exaflow.algorithms.federated.feature_engineering import MinMaxScaler
from exaflow.algorithms.federated.feature_engineering import MissingIndicator
from exaflow.algorithms.federated.feature_engineering import QuantileTransformerApprox
from exaflow.algorithms.federated.feature_engineering import RobustScalerApprox
from exaflow.algorithms.federated.feature_engineering import SimpleImputer
from exaflow.algorithms.federated.feature_engineering import SqrtTransformer
from exaflow.algorithms.federated.feature_engineering import StandardScaler
from exaflow.algorithms.federated.feature_engineering import YeoJohnsonTransformer
from exaflow.algorithms.federated.outliers import OutlierDetector
from exaflow.algorithms.federated.outliers import Winsorizer
from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.utils.interfaces import FederatedTransformer
from tests.standalone_tests.federated_algorithms.utils.dummy_agg_client import (
    DummyAggClient,
)

FACTORIES = [
    SimpleImputer,
    MissingIndicator,
    Log1pTransformer,
    SqrtTransformer,
    AsinhTransformer,
    StandardScaler,
    RobustScalerApprox,
    MinMaxScaler,
    MaxAbsScaler,
    Winsorizer,
    OutlierDetector,
    YeoJohnsonTransformer,
    lambda: QuantileTransformerApprox(n_quantiles=5, platform_minimum=10),
]


@pytest.mark.parametrize("factory", FACTORIES)
def test_named_fit_and_transform_match_the_existing_protocol(factory):
    model = factory()
    assert isinstance(model, FederatedTransformer)
    assert set(inspect.signature(model.fit).parameters) == {
        "agg_client",
        "data",
        "categorical_vars",
        "numerical_vars",
    }
    frame = pd.DataFrame(
        {
            "ignore": ["private"] * 80,
            "b": np.linspace(0, 8, 80),
            "a": np.linspace(2, 4, 80),
        }
    )
    original = frame.copy(deep=True)
    columns = {"categorical_vars": [], "numerical_vars": ["a", "b"]}
    assert model.fit(agg_client=DummyAggClient(), data=frame, **columns) is None
    assert model.get_feature_names_out(**columns) == ["a", "b"]
    result = model.transform(frame, **columns)
    assert result.shape == (80, 2)
    pd.testing.assert_frame_equal(frame, original)
    with pytest.raises(BadInputError, match="names and order"):
        model.transform(frame, categorical_vars=[], numerical_vars=["b", "a"])
    if hasattr(model, "recipe_"):
        state = json.loads(json.dumps(asdict(model.recipe_), allow_nan=False))
        replay = type(model).from_recipe(state)
        np.testing.assert_array_equal(replay.transform(frame, **columns), result)


@pytest.mark.parametrize("factory", FACTORIES)
def test_named_transformers_reject_array_public_inputs(factory):
    with pytest.raises(BadInputError, match="DataFrame"):
        factory().fit(
            agg_client=DummyAggClient(),
            data=np.ones((80, 1)),
            categorical_vars=[],
            numerical_vars=["a"],
        )


def test_federated_pipeline_and_column_transformer_use_the_real_scaler():
    frame = pd.DataFrame({"a": [1.0, 3.0, 5.0], "b": [4.0, 4.0, 4.0]})
    columns = {"categorical_vars": [], "numerical_vars": ["a", "b"]}
    pipeline = FederatedPipeline([("scale", MinMaxScaler())])
    actual = pipeline.fit_transform(agg_client=DummyAggClient(), data=frame, **columns)
    np.testing.assert_array_equal(actual, [[0, 0], [0.5, 0], [1, 0]])
    assert pipeline.get_feature_names_out(**columns) == ["a", "b"]
    grouped = FederatedColumnTransformer(
        [("scale", MinMaxScaler(), ["a"])], remainder="passthrough"
    )
    grouped.fit(agg_client=DummyAggClient(), data=frame, **columns)
    np.testing.assert_array_equal(
        grouped.transform(frame, **columns), [[0, 4], [0.5, 4], [1, 4]]
    )


def test_categorical_mode_uses_named_metadata_vocabulary_and_replays():
    frame = pd.DataFrame({"group": ["a", "b", "a", None], "unused": [9] * 4})
    model = SimpleImputer("most_frequent", categories=[["a", "b"]])
    columns = {"categorical_vars": ["group"], "numerical_vars": []}
    model.fit(agg_client=DummyAggClient(), data=frame, **columns)
    assert model.transform(frame, **columns).tolist() == [["a"], ["b"], ["a"], ["a"]]
    assert model.recipe_.categorical_features == ["group"]
    replay = SimpleImputer.from_recipe(asdict(model.recipe_))
    np.testing.assert_array_equal(
        model.transform(frame, **columns), replay.transform(frame, **columns)
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"feature_names": []},
        {"feature_names": ["a", "a"]},
        {"categorical_features": ["unknown"]},
        {"unknown": 1},
        {"schema_version": "1"},
    ],
)
def test_replay_validates_named_state_without_pydantic(changes):
    frame = pd.DataFrame({"a": [1.0, 2.0]})
    model = StandardScaler()
    model.fit(
        agg_client=DummyAggClient(),
        data=frame,
        categorical_vars=[],
        numerical_vars=["a"],
    )
    state = asdict(model.recipe_)
    state.update(changes)
    with pytest.raises(BadInputError):
        StandardScaler.from_recipe(state)


@pytest.mark.parametrize(
    "categorical,numerical", [(None, ["a"]), ([], "a"), ([], ["a", "a"]), ([], [])]
)
def test_invalid_feature_declarations_fail_with_bad_input(categorical, numerical):
    with pytest.raises(BadInputError):
        StandardScaler().fit(
            agg_client=DummyAggClient(),
            data=pd.DataFrame({"a": [1.0, 2.0]}),
            categorical_vars=categorical,
            numerical_vars=numerical,
        )


def test_constant_categorical_imputation_uses_declared_schema_when_all_values_missing():
    frame = pd.DataFrame({"group": [None, None]})
    model = SimpleImputer("constant", fill_value="missing")
    columns = {"categorical_vars": ["group"], "numerical_vars": []}
    model.fit(agg_client=None, data=frame, **columns)
    assert model.transform(frame, **columns).tolist() == [["missing"], ["missing"]]
    replay = SimpleImputer.from_recipe(asdict(model.recipe_))
    np.testing.assert_array_equal(
        replay.transform(frame, **columns), model.transform(frame, **columns)
    )


def test_numeric_mode_cannot_masquerade_as_declared_numerical_features():
    with pytest.raises(BadInputError, match="metadata vocabulary"):
        SimpleImputer("most_frequent", categories=[[1.37, 8.21]]).fit(
            agg_client=DummyAggClient(),
            data=pd.DataFrame({"value": [1.37, 8.21]}),
            categorical_vars=[],
            numerical_vars=["value"],
        )


@pytest.mark.parametrize("factory", FACTORIES)
def test_nullable_pandas_numerical_input_preserves_source_and_missingness(factory):
    frame = pd.DataFrame({"a": pd.array(range(80), dtype="Int64")})
    frame.loc[3, "a"] = pd.NA
    original = frame.copy(deep=True)
    model = factory()
    columns = {"categorical_vars": [], "numerical_vars": ["a"]}
    model.fit(agg_client=DummyAggClient(), data=frame, **columns)
    actual = model.transform(frame, **columns)
    assert actual.shape == (80, 1)
    if isinstance(model, MissingIndicator):
        assert actual[3, 0] == 1
    elif isinstance(model, OutlierDetector):
        assert not actual[3, 0]
    elif isinstance(model, SimpleImputer):
        assert np.isfinite(actual[3, 0])
    else:
        assert np.isnan(actual[3, 0])
    pd.testing.assert_frame_equal(frame, original)
