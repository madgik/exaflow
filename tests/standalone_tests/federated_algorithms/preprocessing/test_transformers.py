from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from scipy.stats import yeojohnson
from sklearn.preprocessing import MaxAbsScaler as SklearnMaxAbsScaler
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

from exaflow.algorithms.federated.preprocessing import AsinhTransformer
from exaflow.algorithms.federated.preprocessing import Log1pTransformer
from exaflow.algorithms.federated.preprocessing import MaxAbsScaler
from exaflow.algorithms.federated.preprocessing import MinMaxScaler
from exaflow.algorithms.federated.preprocessing import MissingIndicator
from exaflow.algorithms.federated.preprocessing import OutlierDetector
from exaflow.algorithms.federated.preprocessing import QuantileTransformerApprox
from exaflow.algorithms.federated.preprocessing import RobustScalerApprox
from exaflow.algorithms.federated.preprocessing import SimpleImputer
from exaflow.algorithms.federated.preprocessing import SqrtTransformer
from exaflow.algorithms.federated.preprocessing import StandardScaler
from exaflow.algorithms.federated.preprocessing import Winsorizer
from exaflow.algorithms.federated.preprocessing import YeoJohnsonTransformer
from exaflow.algorithms.federated.preprocessing.recipes import QuantileTransformerRecipe
from exaflow.algorithms.federated.preprocessing.recipes import ScalerRecipe
from exaflow.algorithms.federated.preprocessing.recipes import YeoJohnsonRecipe
from exaflow.algorithms.federated.utils import BadInputError
from tests.standalone_tests.federated_algorithms.utils.dummy_agg_client import (
    DummyAggClient,
)
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    AggregationCoordinator,
)
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    SimulatedAggClient,
)


def _fit_in_workers(factory, parts):
    coordinator = AggregationCoordinator(len(parts))

    def fit_one(item):
        worker_id, part = item
        transformer = factory()
        transformer.fit(
            part,
            agg_client=SimulatedAggClient(worker_id=worker_id, coordinator=coordinator),
        )
        return transformer

    with ThreadPoolExecutor(max_workers=len(parts)) as executor:
        return list(executor.map(fit_one, enumerate(parts)))


def test_simple_imputer_mean_is_global_and_does_not_mutate_inputs():
    left = np.array([[1.0], [np.nan]])
    right = np.array([[5.0], [9.0]])
    original_left = left.copy()
    transformers = _fit_in_workers(lambda: SimpleImputer("mean"), [left, right])

    assert np.allclose(transformers[0].statistics_, [5.0])
    assert np.allclose(transformers[1].transform(right), right)
    assert np.array_equal(left, original_left, equal_nan=True)
    assert np.allclose(transformers[0].transform(left), [[1.0], [5.0]])


def test_simple_imputer_median_constant_and_safe_categorical_mode():
    data = np.array([[1.0], [2.0], [100.0], [np.nan]])
    median = SimpleImputer("median").fit(data, agg_client=DummyAggClient())
    constant = SimpleImputer("constant", fill_value=-1).fit(
        data, agg_client=DummyAggClient()
    )
    categorical = SimpleImputer("most_frequent").fit(
        np.array([["a"], ["b"], ["a"], [None]], dtype=object),
        agg_client=DummyAggClient(),
    )

    assert median.statistics_[0] == pytest.approx(2.0)
    assert np.array_equal(constant.transform(data), [[1.0], [2.0], [100.0], [-1.0]])
    assert categorical.transform(np.array([[None]], dtype=object)).tolist() == [["a"]]


def test_simple_imputer_rejects_numerical_mode_without_safe_protocol():
    with pytest.raises(BadInputError, match="numeric mode"):
        SimpleImputer("most_frequent").fit(
            np.array([[1.0], [2.0], [np.nan]]), agg_client=DummyAggClient()
        )


def test_simple_imputer_mode_rejects_globally_all_missing_feature():
    with pytest.raises(BadInputError, match="no valid observations"):
        SimpleImputer("most_frequent").fit(
            np.array([[None], [None]], dtype=object),
            agg_client=DummyAggClient(),
        )


def test_missing_indicator_is_parameter_free_and_preserves_shape():
    data = np.array([[1.0, np.nan], [np.nan, 2.0]])
    original = data.copy()
    indicator = MissingIndicator().fit(data)

    assert indicator.transform(data).tolist() == [[0, 1], [1, 0]]
    assert np.array_equal(data, original, equal_nan=True)
    assert MissingIndicator().fit(np.array([1.0, np.nan])).transform(
        np.array([1.0, np.nan])
    ).tolist() == [0, 1]


@pytest.mark.parametrize(
    "transformer, expected",
    [
        (Log1pTransformer(), np.log1p([0.0, 1.0, 9.0])),
        (SqrtTransformer(), np.sqrt([0.0, 1.0, 9.0])),
        (AsinhTransformer(), np.arcsinh([-9.0, 0.0, 9.0])),
    ],
)
def test_parameter_transformers_match_numpy_and_do_not_mutate(transformer, expected):
    data = np.array([0.0, 1.0, 9.0])
    if isinstance(transformer, AsinhTransformer):
        data = np.array([-9.0, 0.0, 9.0])
    original = data.copy()
    transformer.fit(data, agg_client=DummyAggClient())

    assert np.allclose(transformer.transform(data), expected)
    assert np.array_equal(data, original)


@pytest.mark.parametrize("transformer", [Log1pTransformer(), SqrtTransformer()])
def test_domain_validation_is_global(transformer):
    parts = [np.array([[1.0], [2.0]]), np.array([[-1.0], [3.0]])]
    with pytest.raises(BadInputError, match="non-negative"):
        _fit_in_workers(lambda: type(transformer)(), parts)


def test_standard_scaler_uses_two_pass_global_variance():
    parts = [np.array([[1e12], [1e12 + 1.0]]), np.array([[1e12 + 2.0]])]
    fitted = _fit_in_workers(StandardScaler, parts)
    pooled = np.vstack(parts)
    reference = SklearnStandardScaler().fit(pooled).transform(pooled)
    actual = np.vstack([fitted[0].transform(parts[0]), fitted[1].transform(parts[1])])

    assert np.allclose(actual, reference, rtol=1e-10, atol=1e-8)
    assert fitted[0].recipe_.model_dump_json()
    replay = StandardScaler.from_recipe(
        ScalerRecipe.model_validate_json(fitted[0].recipe_.model_dump_json())
    )
    assert np.allclose(replay.transform(pooled), actual)


def test_minmax_scaler_matches_sklearn_and_replays_from_json_recipe():
    parts = [np.array([[-2.0], [0.0]]), np.array([[4.0], [10.0]])]
    fitted = _fit_in_workers(lambda: MinMaxScaler(feature_range=(-1.0, 2.0)), parts)
    pooled = np.vstack(parts)
    reference = SklearnMinMaxScaler(feature_range=(-1.0, 2.0)).fit(pooled)
    actual = fitted[0].transform(pooled)
    recipe = ScalerRecipe.model_validate_json(fitted[0].recipe_.model_dump_json())
    replay = MinMaxScaler.from_recipe(recipe)

    assert np.allclose(actual, reference.transform(pooled))
    assert np.allclose(replay.transform(pooled), actual)


def test_minmax_and_maxabs_handle_constant_features_deterministically():
    data = np.array([[3.0, -2.0], [3.0, 4.0]])
    minimum = MinMaxScaler().fit(data, agg_client=DummyAggClient())
    maximum = MaxAbsScaler().fit(data, agg_client=DummyAggClient())

    assert np.allclose(minimum.transform(data)[:, 0], 0.0)
    assert np.allclose(maximum.transform(data)[:, 0], 1.0)
    assert np.allclose(maximum.transform(data)[:, 1], [-0.5, 1.0])


def test_maxabs_rejects_feature_without_valid_values():
    with pytest.raises(BadInputError, match="without valid values"):
        MaxAbsScaler().fit(np.array([[np.nan], [np.nan]]), agg_client=DummyAggClient())


def test_winsorizer_and_robust_scaler_use_global_percentiles():
    parts = [np.array([[0.0], [1.0], [2.0]]), np.array([[3.0], [4.0], [100.0]])]
    winsorized = _fit_in_workers(lambda: Winsorizer(0.0, 0.8), parts)
    robust = _fit_in_workers(RobustScalerApprox, parts)
    pooled = np.vstack(parts)

    assert np.max(winsorized[0].transform(pooled)) <= 100.0
    assert np.all(np.isfinite(robust[1].transform(pooled)))
    assert np.array_equal(parts[0], [[0.0], [1.0], [2.0]])


def test_outlier_detector_is_diagnostic_only():
    data = np.array([[0.0], [0.0], [0.0], [10.0]])
    detector = OutlierDetector(threshold=1.5).fit(data, agg_client=DummyAggClient())

    mask = detector.transform(data)
    assert mask.dtype == bool
    assert mask[-1]
    assert np.array_equal(data, [[0.0], [0.0], [0.0], [10.0]])


def test_yeo_johnson_supports_negative_and_positive_values():
    data = np.array([[-3.0], [-1.0], [0.0], [2.0], [5.0]])
    transformer = YeoJohnsonTransformer().fit(data, agg_client=DummyAggClient())

    assert np.all(np.isfinite(transformer.transform(data)))
    assert transformer.recipe_.lambdas[0] == pytest.approx(transformer.lambdas_[0])
    replay = YeoJohnsonTransformer.from_recipe(
        YeoJohnsonRecipe.model_validate_json(transformer.recipe_.model_dump_json())
    )
    assert np.allclose(replay.transform(data), transformer.transform(data))


def test_yeo_johnson_is_partition_stable_and_matches_scipy_reference():
    pooled = np.array([[-4.0], [-2.0], [-1.0], [0.0], [1.0], [3.0], [8.0]])
    fitted = _fit_in_workers(
        YeoJohnsonTransformer, [pooled[:2], pooled[2:4], pooled[4:]]
    )
    expected, expected_lambda = yeojohnson(pooled[:, 0])

    assert np.allclose(fitted[0].lambdas_, fitted[1].lambdas_)
    assert np.allclose(fitted[1].lambdas_, fitted[2].lambdas_)
    assert fitted[0].lambdas_[0] == pytest.approx(expected_lambda, abs=1e-3)
    actual = np.vstack(
        [
            fitted[0].transform(pooled[:2]),
            fitted[1].transform(pooled[2:4]),
            fitted[2].transform(pooled[4:]),
        ]
    )
    assert np.allclose(actual[:, 0], expected, atol=1e-3)


def test_quantile_transformer_is_monotone_and_replays_recipe():
    data = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    transformer = QuantileTransformerApprox(n_quantiles=5).fit(
        data, agg_client=DummyAggClient()
    )
    transformed = transformer.transform(data)

    assert np.all(np.diff(transformed[:, 0]) >= 0)
    assert transformed[0, 0] == pytest.approx(0.0)
    assert transformed[-1, 0] == pytest.approx(1.0)
    recipe = QuantileTransformerRecipe.model_validate_json(
        transformer.recipe_.model_dump_json()
    )
    replay = QuantileTransformerApprox.from_recipe(recipe)
    assert np.allclose(replay.transform(data), transformed)


def test_quantile_transformer_is_partition_stable():
    pooled = np.arange(20.0).reshape(-1, 1)
    fitted = _fit_in_workers(
        lambda: QuantileTransformerApprox(n_quantiles=5),
        [pooled[:3], pooled[3:17], pooled[17:]],
    )
    actual = np.vstack(
        [
            fitted[0].transform(pooled[:3]),
            fitted[1].transform(pooled[3:17]),
            fitted[2].transform(pooled[17:]),
        ]
    )
    assert np.all(np.diff(actual[:, 0]) >= 0)
    assert actual[0, 0] == pytest.approx(0.0)
    assert actual[-1, 0] == pytest.approx(1.0)
