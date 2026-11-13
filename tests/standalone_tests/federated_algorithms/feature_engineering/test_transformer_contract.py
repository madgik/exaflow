"""Numerical regressions retained through the named-column interface."""

import json
from dataclasses import asdict

import numpy as np
import pytest

from exaflow.algorithms.federated.feature_engineering.asinh_transformer import (
    AsinhTransformer,
)
from exaflow.algorithms.federated.feature_engineering.log1p_transformer import (
    Log1pTransformer,
)
from exaflow.algorithms.federated.feature_engineering.maxabs_scaler import MaxAbsScaler
from exaflow.algorithms.federated.feature_engineering.minmax_scaler import MinMaxScaler
from exaflow.algorithms.federated.feature_engineering.missing_indicator import (
    MissingIndicator,
)
from exaflow.algorithms.federated.feature_engineering.quantile_transformer import (
    QuantileTransformerApprox,
)
from exaflow.algorithms.federated.feature_engineering.robust_scaler import (
    RobustScalerApprox,
)
from exaflow.algorithms.federated.feature_engineering.simple_imputer import (
    SimpleImputer,
)
from exaflow.algorithms.federated.feature_engineering.sqrt_transformer import (
    SqrtTransformer,
)
from exaflow.algorithms.federated.feature_engineering.standard_scaler import (
    StandardScaler,
)
from exaflow.algorithms.federated.feature_engineering.yeo_johnson_transformer import (
    YeoJohnsonTransformer,
)
from exaflow.algorithms.federated.outliers.outlier_detector import OutlierDetector
from exaflow.algorithms.federated.outliers.winsorizer import Winsorizer
from exaflow.algorithms.federated.utils import BadInputError
from tests.standalone_tests.federated_algorithms.feature_engineering.transformation_test_support import (
    fit as _fit,
)
from tests.standalone_tests.federated_algorithms.feature_engineering.transformation_test_support import (
    transform as _transform,
)
from tests.standalone_tests.federated_algorithms.utils.dummy_agg_client import (
    DummyAggClient,
)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SimpleImputer("mean"),
        lambda: SimpleImputer("median"),
        lambda: SimpleImputer("constant"),
        RobustScalerApprox,
        MaxAbsScaler,
        Winsorizer,
        OutlierDetector,
        lambda: OutlierDetector(mode="robust"),
    ],
)
def test_fitted_recipe_json_roundtrip_preserves_transform(factory):
    data = np.array([[0.0], [1.0], [3.0], [10.0], [np.nan]])
    model = _fit(factory(), data, agg_client=DummyAggClient())
    replay = type(model).from_recipe(
        json.loads(json.dumps(asdict(model.recipe_), allow_nan=False))
    )
    np.testing.assert_allclose(
        _transform(replay, data), _transform(model, data), equal_nan=True
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SimpleImputer(strategy=[]),
        lambda: MinMaxScaler((1, 1)),
        lambda: MinMaxScaler((np.nan, 1)),
        lambda: Winsorizer("a", 0.9),
        lambda: Winsorizer(0.9, 0.1),
        lambda: OutlierDetector(threshold=np.inf),
        lambda: OutlierDetector(mode=[]),
        lambda: QuantileTransformerApprox(n_quantiles=np.nan, platform_minimum=10),
        lambda: QuantileTransformerApprox(n_quantiles=True, platform_minimum=10),
        lambda: QuantileTransformerApprox(platform_minimum=0),
        lambda: QuantileTransformerApprox(
            platform_minimum=10, min_observations_per_interval="1"
        ),
        lambda: QuantileTransformerApprox(platform_minimum=10, n_bins=1),
    ],
)
def test_constructor_errors_are_bad_input(factory):
    with pytest.raises(BadInputError):
        factory()


@pytest.mark.parametrize(
    "factory",
    [
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
        lambda: QuantileTransformerApprox(platform_minimum=10),
    ],
)
def test_unfitted_transform_raises_bad_input(factory):
    with pytest.raises(BadInputError, match="fitted"):
        _transform(factory(), [[1.0]])


@pytest.mark.parametrize(
    "factory",
    [
        Log1pTransformer,
        SqrtTransformer,
        AsinhTransformer,
        StandardScaler,
        RobustScalerApprox,
        MinMaxScaler,
        MaxAbsScaler,
        Winsorizer,
        YeoJohnsonTransformer,
        lambda: QuantileTransformerApprox(n_quantiles=5, platform_minimum=10),
    ],
)
def test_numeric_transform_preserves_nan_shape_and_input(factory):
    train = np.linspace(0, 5, 40).reshape(-1, 1)
    model = _fit(factory(), train, agg_client=DummyAggClient())
    data = np.array([0, np.nan, 5])
    copy = data.copy()
    actual = _transform(model, data)
    assert actual.shape == data.shape
    assert np.isnan(actual[1])
    np.testing.assert_array_equal(data, copy)
