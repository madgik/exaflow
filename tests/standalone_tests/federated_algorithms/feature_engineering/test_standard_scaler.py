"""Fitted-object lifecycle, pooled mathematics and native composition."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler as ReferenceScaler

from exaflow.algorithms.federated import FederatedStandardScaler
from exaflow.algorithms.federated.compose import FederatedColumnTransformer
from exaflow.algorithms.federated.linear_model import FederatedOLS
from exaflow.algorithms.federated.pipeline import FederatedPipeline
from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.utils.interfaces import FederatedTransformer
from tests.standalone_tests.federated_algorithms.utils.dummy_agg_client import (
    DummyAggClient,
)
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    AggregationCoordinator,
)
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    SimulatedAggClient,
)


def fit(data, model=None, client=None):
    model = model if model is not None else FederatedStandardScaler()
    model.fit(
        agg_client=client if client is not None else DummyAggClient(),
        data=data,
        categorical_vars=[],
        numerical_vars=list(data.columns),
    )
    return model


def transform(model, data):
    return model.transform(data, categorical_vars=[], numerical_vars=list(data.columns))


def fit_partitions(parts):
    coordinator = AggregationCoordinator(len(parts))

    def fit_one(item):
        worker_id, part = item
        try:
            return fit(
                part,
                client=SimulatedAggClient(worker_id=worker_id, coordinator=coordinator),
            )
        except Exception as exc:
            coordinator.abort(exc)
            raise

    with ThreadPoolExecutor(max_workers=len(parts)) as executor:
        futures = [executor.submit(fit_one, item) for item in enumerate(parts)]
        return [future.result(timeout=30) for future in futures]


def test_fit_returns_none_and_transform_does_not_refit():
    class CountingClient(DummyAggClient):
        calls = 0

        def sum(self, value):
            self.calls += 1
            return super().sum(value)

    data = pd.DataFrame({"x": [1.0, 3.0, 5.0]})
    original = data.copy(deep=True)
    client = CountingClient()
    model = FederatedStandardScaler()
    assert (
        model.fit(
            agg_client=client, data=data, categorical_vars=[], numerical_vars=["x"]
        )
        is None
    )
    assert client.calls == 3
    assert isinstance(model, FederatedTransformer)
    assert not hasattr(model, "recipe_")
    assert not hasattr(model, "from_recipe")
    mean, scale = model.mean_.copy(), model.scale_.copy()
    new = pd.DataFrame({"x": [9.0, np.nan, 15.0]})
    np.testing.assert_allclose(
        transform(model, new),
        ReferenceScaler().fit(data).transform(new),
        equal_nan=True,
    )
    np.testing.assert_array_equal(model.mean_, mean)
    np.testing.assert_array_equal(model.scale_, scale)
    assert client.calls == 3
    pd.testing.assert_frame_equal(data, original)


def test_uneven_partitions_use_global_two_pass_statistics():
    data = pd.DataFrame({"x": [1e12, 1e12 + 1, 1e12 + 2], "y": [1, np.nan, 9]})
    parts = [data.iloc[:2], data.iloc[2:]]
    models = fit_partitions(parts)
    reference = ReferenceScaler().fit(data)
    for model in models:
        np.testing.assert_allclose(model.mean_, reference.mean_)
        np.testing.assert_allclose(model.scale_, reference.scale_)
        np.testing.assert_allclose(
            transform(model, data),
            reference.transform(data),
            rtol=1e-10,
            atol=1e-8,
            equal_nan=True,
        )


def test_all_missing_local_feature_uses_other_sites_statistics():
    left = pd.DataFrame({"x": [np.nan, np.nan], "y": [1.0, np.nan]})
    right = pd.DataFrame({"x": [2.0, 4.0, 6.0], "y": [3.0, 5.0, 7.0]})
    pooled = pd.concat([left, right], ignore_index=True)
    reference = ReferenceScaler().fit(pooled)
    for model in fit_partitions([left, right]):
        np.testing.assert_allclose(model.mean_, reference.mean_)
        np.testing.assert_allclose(model.scale_, reference.scale_)
        np.testing.assert_allclose(
            transform(model, pooled), reference.transform(pooled), equal_nan=True
        )
        assert np.isnan(transform(model, left)[:, 0]).all()


def test_empty_sites_contribute_neutral_statistics():
    empty = pd.DataFrame({"x": pd.Series(dtype=float)})
    populated = pd.DataFrame({"x": [1.0, 3.0, 5.0]})
    models = fit_partitions([empty, populated, empty.copy()])
    reference = ReferenceScaler().fit(populated)
    for model in models:
        np.testing.assert_allclose(model.mean_, reference.mean_)
        np.testing.assert_allclose(model.scale_, reference.scale_)
        assert transform(model, empty).shape == (0, 1)
        np.testing.assert_allclose(
            transform(model, populated), reference.transform(populated)
        )


def test_globally_empty_population_is_still_rejected():
    empty = pd.DataFrame({"x": pd.Series(dtype=float)})
    with pytest.raises(BadInputError, match="no valid observations"):
        fit_partitions([empty, empty.copy()])


@pytest.mark.parametrize("values", [[0.1, 0.1, 0.1], [1e-17, 2e-17, 3e-17]])
def test_constant_and_tiny_magnitude_features_match_reference(values):
    data = pd.DataFrame({"x": values})
    np.testing.assert_allclose(
        transform(fit(data), data), ReferenceScaler().fit_transform(data), atol=1e-14
    )


def test_many_constant_rows_and_sites():
    data = pd.DataFrame(np.tile(np.linspace(0.01, 100, 200), (10000, 1)))
    data.columns = [f"x{i}" for i in range(200)]
    models = fit_partitions(
        [data.iloc[i : i + 1000] for i in range(0, len(data), 1000)]
    )
    for model in models:
        np.testing.assert_array_equal(model.scale_, np.ones(200))
        np.testing.assert_allclose(transform(model, data.iloc[:1]), 0, atol=1e-10)


def test_fit_transform_and_nan_preservation():
    train = pd.DataFrame({"x": [1.0, np.nan, 5.0], "y": [2.0, 4.0, 8.0]})
    original = train.copy(deep=True)
    model = FederatedStandardScaler()
    result = model.fit_transform(
        agg_client=DummyAggClient(),
        data=train,
        categorical_vars=[],
        numerical_vars=list(train),
    )
    np.testing.assert_allclose(result, transform(fit(train), train), equal_nan=True)
    assert result.shape == train.shape
    assert np.isnan(result[1, 0])
    pd.testing.assert_frame_equal(train, original)


def test_unfitted_transform_rejected():
    with pytest.raises(BadInputError, match="not fitted"):
        transform(FederatedStandardScaler(), pd.DataFrame({"x": [1.0]}))


@pytest.mark.parametrize(
    "data",
    [
        pd.DataFrame({"x": [np.nan]}),
        pd.DataFrame({"x": [np.inf]}),
        pd.DataFrame({"x": ["bad"]}),
        pd.DataFrame({"x": [1.0, 2.0], "missing": [np.nan, np.nan]}),
    ],
)
def test_invalid_fit_inputs_leave_model_unfitted(data):
    model = FederatedStandardScaler()
    with pytest.raises(BadInputError):
        fit(data, model)
    assert not hasattr(model, "feature_names_in_")
    assert not hasattr(model, "mean_")


def test_nonfinite_aggregated_parameters_do_not_publish_fitted_state():
    class NonfiniteSumClient(DummyAggClient):
        calls = 0

        def sum(self, value):
            self.calls += 1
            if self.calls == 1:
                return np.array([np.inf])
            return super().sum(value)

    model = FederatedStandardScaler()
    with np.errstate(invalid="ignore", over="ignore"):
        with pytest.raises(BadInputError, match="Fitted parameters"):
            fit(pd.DataFrame({"x": [1.0, 2.0]}), model, NonfiniteSumClient())
    assert not hasattr(model, "mean_")
    assert not hasattr(model, "scale_")
    assert not hasattr(model, "feature_names_in_")


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("mean_", [np.nan]),
        ("mean_", [np.inf]),
        ("mean_", [[1.0]]),
        ("scale_", [0]),
        ("scale_", [-1]),
        ("scale_", [np.nan]),
        ("scale_", [np.inf]),
        ("scale_", [1, 2]),
    ],
)
def test_invalid_fitted_attributes_rejected(attribute, value):
    data = pd.DataFrame({"x": [1.0, 2.0]})
    model = fit(data)
    setattr(model, attribute, value)
    with pytest.raises(BadInputError, match="Fitted parameters"):
        transform(model, data)


def test_schema_and_input_validation():
    data = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    model = fit(data)
    for changed in (data[["y", "x"]], data[["x"]], data.rename(columns={"x": "z"})):
        with pytest.raises(BadInputError, match="names and order"):
            transform(model, changed)
    with pytest.raises(BadInputError, match="required columns"):
        model.transform(data[["x"]], categorical_vars=[], numerical_vars=["x", "y"])
    with pytest.raises(BadInputError, match="DataFrame"):
        model.transform(data.to_numpy(), categorical_vars=[], numerical_vars=["x", "y"])
    with pytest.raises(BadInputError, match="distinct|unique"):
        fit(pd.DataFrame([[1, 2]], columns=["x", "x"]))
    with pytest.raises(BadInputError, match="numerical variables only"):
        model.fit(
            agg_client=DummyAggClient(),
            data=data,
            categorical_vars=["x"],
            numerical_vars=["y"],
        )
    with pytest.raises(BadInputError, match="infinite"):
        transform(model, data * np.inf)


def test_refit_replaces_complete_state_and_failed_refit_does_not_mix_schemas():
    model = fit(pd.DataFrame({"x": [1.0, 3.0]}))
    with pytest.raises(BadInputError):
        fit(pd.DataFrame({"different": [np.nan]}), model)
    assert model.feature_names_in_ == ["x"]
    np.testing.assert_allclose(transform(model, pd.DataFrame({"x": [3.0]})), [[1]])
    assert (
        model.fit(
            agg_client=DummyAggClient(),
            data=pd.DataFrame({"y": [10.0, 20.0]}),
            categorical_vars=[],
            numerical_vars=["y"],
        )
        is None
    )
    assert model.feature_names_in_ == ["y"]
    np.testing.assert_allclose(model.mean_, [15])


def test_transform_selects_named_features_with_extra_columns_and_empty_rows():
    train = pd.DataFrame({"x": [1.0, 3.0], "y": [2.0, 6.0]})
    model = fit(train)
    new = pd.DataFrame({"unused": ["text"], "y": [10.0], "x": [5.0]})
    original = new.copy(deep=True)
    actual = model.transform(new, categorical_vars=[], numerical_vars=["x", "y"])
    np.testing.assert_allclose(
        actual, ReferenceScaler().fit(train).transform(new[["x", "y"]])
    )
    empty = model.transform(
        new.iloc[:0], categorical_vars=[], numerical_vars=["x", "y"]
    )
    assert empty.shape == (0, 2)
    pd.testing.assert_frame_equal(new, original)


def test_existing_pipeline_and_column_transformer_accept_scaler():
    data = pd.DataFrame({"x": [1.0, 3.0, 5.0], "keep": [2.0, 7.0, 9.0]})
    pipeline = FederatedPipeline([("scaler", FederatedStandardScaler())])
    pipeline.fit(
        agg_client=DummyAggClient(),
        data=data,
        categorical_vars=[],
        numerical_vars=["x"],
    )
    np.testing.assert_allclose(
        pipeline.transform(data, categorical_vars=[], numerical_vars=["x"]),
        ReferenceScaler().fit_transform(data[["x"]]),
    )
    columns = FederatedColumnTransformer(
        [("scale", FederatedStandardScaler(), ["x"])], remainder="passthrough"
    )
    columns.fit(
        agg_client=DummyAggClient(),
        data=data,
        categorical_vars=[],
        numerical_vars=["x", "keep"],
    )
    expected = np.column_stack(
        [ReferenceScaler().fit_transform(data[["x"]]), data["keep"]]
    )
    np.testing.assert_allclose(
        columns.transform(data, categorical_vars=[], numerical_vars=["x", "keep"]),
        expected,
    )


def test_existing_pipeline_passes_scaled_features_to_final_estimator():
    data = pd.DataFrame({"x": np.arange(1.0, 7.0)})
    target = np.array([7.1, 9.0, 10.9, 13.1, 15.0, 16.9])
    pipeline = FederatedPipeline(
        [
            ("scale", FederatedStandardScaler()),
            ("model", FederatedOLS(fit_intercept=True)),
        ]
    )
    result = pipeline.fit(
        agg_client=DummyAggClient(),
        data=data,
        y=target,
        categorical_vars=[],
        numerical_vars=["x"],
    )
    scaled = ReferenceScaler().fit_transform(data)
    design = np.column_stack([np.ones(len(data)), scaled])
    expected, *_ = np.linalg.lstsq(design, target, rcond=None)
    np.testing.assert_allclose(result.params, expected)
    assert result.nobs == len(data)
