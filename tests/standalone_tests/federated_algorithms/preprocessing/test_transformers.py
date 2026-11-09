import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from pydantic import ValidationError
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
        try:
            transformer.fit(
                part,
                agg_client=SimulatedAggClient(
                    worker_id=worker_id, coordinator=coordinator
                ),
            )
        except Exception as exc:
            coordinator.abort(exc)
            raise
        return transformer

    with ThreadPoolExecutor(max_workers=len(parts)) as executor:
        futures = [executor.submit(fit_one, item) for item in enumerate(parts)]
        try:
            return [future.result(timeout=30) for future in futures]
        except Exception as exc:
            coordinator.abort(exc)
            raise


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
    categorical = SimpleImputer("most_frequent", categories=[["a", "b"]]).fit(
        np.array([["a"], ["b"], ["a"], [None]], dtype=object),
        agg_client=DummyAggClient(),
    )

    assert median.statistics_[0] == pytest.approx(2.0)
    assert np.array_equal(constant.transform(data), [[1.0], [2.0], [100.0], [-1.0]])
    assert categorical.transform(np.array([[None]], dtype=object)).tolist() == [["a"]]


def test_simple_imputer_constant_defaults_to_zero_for_numeric_input():
    data = np.array([[1.0], [np.nan]])
    imputer = SimpleImputer("constant").fit(data, agg_client=DummyAggClient())

    assert np.array_equal(imputer.transform(data), [[1.0], [0.0]])


def test_simple_imputer_rejects_numerical_mode_without_safe_protocol():
    with pytest.raises(BadInputError, match="metadata vocabulary"):
        SimpleImputer("most_frequent").fit(
            np.array([[1.0], [2.0], [np.nan]]), agg_client=DummyAggClient()
        )


def test_simple_imputer_rejects_numeric_values_stored_as_object():
    with pytest.raises(BadInputError, match="metadata vocabulary"):
        SimpleImputer("most_frequent").fit(
            np.array([[1.37], [8.21], [1.37]], dtype=object),
            agg_client=DummyAggClient(),
        )


def test_simple_imputer_mode_rejects_globally_all_missing_feature():
    with pytest.raises(BadInputError, match="no valid observations"):
        SimpleImputer("most_frequent", categories=[["a", "b"]]).fit(
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


@pytest.mark.parametrize(
    "transformer, invalid",
    [(Log1pTransformer(), np.array([-0.5])), (SqrtTransformer(), np.array([-1.0]))],
)
def test_parameter_transformers_validate_transform_domain(transformer, invalid):
    transformer.fit(np.array([[0.0]]), agg_client=DummyAggClient())

    with pytest.raises(BadInputError, match="requires non-negative"):
        transformer.transform(invalid)


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


def test_standard_scaler_treats_rounding_only_scale_as_constant():
    data = np.array([[0.1], [0.1], [0.1]])
    scaler = StandardScaler().fit(data, agg_client=DummyAggClient())

    assert np.allclose(scaler.transform(data), 0.0)


def test_outlier_detector_treats_rounding_only_scale_as_constant():
    data = np.array([[0.1], [0.1], [0.1]])
    detector = OutlierDetector().fit(data, agg_client=DummyAggClient())

    assert not detector.transform(data).any()


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


def test_minmax_preserves_missing_values_in_constant_features():
    fitted = MinMaxScaler().fit(
        np.array([[3.0], [3.0], [np.nan]]), agg_client=DummyAggClient()
    )

    transformed = fitted.transform(np.array([[3.0], [np.nan], [9.0]]))

    assert np.array_equal(transformed, [[0.0], [np.nan], [6.0]], equal_nan=True)


def test_maxabs_rejects_feature_without_valid_values():
    with pytest.raises(BadInputError, match="without valid values"):
        MaxAbsScaler().fit(np.array([[np.nan], [np.nan]]), agg_client=DummyAggClient())


def test_winsorizer_clips_to_global_fitted_bounds():
    parts = [np.array([[0.0], [1.0], [2.0]]), np.array([[3.0], [4.0], [100.0]])]
    winsorized = _fit_in_workers(lambda: Winsorizer(0.0, 0.8), parts)
    pooled = np.vstack(parts)

    clipped = winsorized[0].transform(pooled)
    expected = np.clip(pooled, winsorized[0].lower_bounds_, winsorized[0].upper_bounds_)
    np.testing.assert_allclose(clipped, expected)
    assert clipped[-1, 0] == winsorized[0].upper_bounds_[0]
    assert clipped[-1, 0] < pooled[-1, 0]
    assert np.array_equal(parts[0], [[0.0], [1.0], [2.0]])


def test_robust_scaler_matches_reference_with_stated_quantile_tolerance():
    pooled = np.linspace(0, 10, 1001).reshape(-1, 1)
    fitted = _fit_in_workers(RobustScalerApprox, [pooled[:100], pooled[100:]])
    q25, median, q75 = np.quantile(pooled[:, 0], [0.25, 0.5, 0.75])
    np.testing.assert_allclose(fitted[0].center_, [median], atol=0.02)
    np.testing.assert_allclose(fitted[0].scale_, [q75 - q25], atol=0.02)
    np.testing.assert_allclose(
        fitted[1].transform(pooled), (pooled - median) / (q75 - q25), atol=0.02
    )


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


def test_yeo_johnson_uses_global_constant_detection_across_partitions():
    parts = [
        np.array([[7.0, 1.0], [7.0, 2.0]]),
        np.array([[1.0, 9.0], [2.0, 9.0]]),
    ]
    fitted = _fit_in_workers(YeoJohnsonTransformer, parts)

    assert np.allclose(fitted[0].lambdas_, fitted[1].lambdas_)


def test_yeo_johnson_all_missing_local_partition_still_joins_aggregation():
    fitted = _fit_in_workers(
        YeoJohnsonTransformer,
        [np.array([[np.nan], [np.nan]]), np.array([[1.0], [2.0]])],
    )

    assert np.allclose(fitted[0].lambdas_, fitted[1].lambdas_)


def test_quantile_transformer_is_monotone_and_replays_recipe():
    data = np.arange(40.0).reshape(-1, 1)
    transformer = QuantileTransformerApprox(n_quantiles=5, platform_minimum=10).fit(
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
    pooled = np.arange(40.0).reshape(-1, 1)
    fitted = _fit_in_workers(
        lambda: QuantileTransformerApprox(n_quantiles=5, platform_minimum=10),
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


def test_quantile_transformer_requires_global_observations_per_interval():
    with pytest.raises(BadInputError, match="global observations"):
        QuantileTransformerApprox(n_quantiles=5, platform_minimum=10).fit(
            np.arange(39.0).reshape(-1, 1), agg_client=DummyAggClient()
        )


def test_quantile_transformer_constant_normal_output_is_centered():
    transformer = QuantileTransformerApprox(
        n_quantiles=5,
        output_distribution="normal",
        platform_minimum=10,
    ).fit(np.ones((40, 1)), agg_client=DummyAggClient())

    assert np.allclose(transformer.transform(np.ones((2, 1))), 0.0)


@pytest.mark.parametrize("start, stop", [(1e6, 1e6 + 1), (1e8, 1e8 + 100), (0, 1e-9)])
def test_quantile_transformer_does_not_flatten_nonconstant_features(start, stop):
    data = np.linspace(start, stop, 100).reshape(-1, 1)
    fitted = QuantileTransformerApprox(n_quantiles=5, platform_minimum=10).fit(
        data, agg_client=DummyAggClient()
    )
    actual = fitted.transform(data)[:, 0]
    assert actual[0] == pytest.approx(0)
    assert actual[-1] == pytest.approx(1)
    assert np.all(np.diff(actual) >= 0)
    assert len(np.unique(actual)) > 2


def test_standard_scaler_preserves_real_variation_at_tiny_magnitudes():
    data = np.array([1e-17, 2e-17, 3e-17]).reshape(-1, 1)
    fitted = StandardScaler().fit(data, agg_client=DummyAggClient())
    expected = SklearnStandardScaler().fit_transform(data)
    np.testing.assert_allclose(fitted.transform(data), expected, atol=1e-14)


def test_constant_imputer_numeric_output_supports_numpy_and_json_replay():
    data = np.array([[1.0], [np.nan]])
    fitted = SimpleImputer("constant").fit(data, agg_client=DummyAggClient())
    for model in (fitted, SimpleImputer.from_recipe(fitted.recipe_.model_dump())):
        actual = model.transform(data)
        assert actual.dtype.kind == "f"
        assert not np.isnan(actual).any()
        np.testing.assert_array_equal(actual, [[1], [0]])
    assert fitted.fill_value is None


def test_quantile_fit_uses_bounded_aggregation_rounds():
    class CountingClient(DummyAggClient):
        def __init__(self):
            self.calls = 0

        def sum(self, values):
            self.calls += 1
            return super().sum(values)

        def min(self, values):
            self.calls += 1
            return super().min(values)

        def max(self, values):
            self.calls += 1
            return super().max(values)

    data = np.random.default_rng(12).normal(size=(1000, 4))
    client = CountingClient()
    QuantileTransformerApprox(n_quantiles=100, platform_minimum=10).fit(
        data, agg_client=client
    )
    # Shared setup plus at most three vector exchanges per histogram level,
    # independent of how many target quantiles occupy those bins.
    assert client.calls <= 6 + 3 * data.shape[1] * 3


@pytest.mark.parametrize(
    "configured, platform, n, passes",
    [
        (1, 10, 39, False),
        (1, 10, 40, True),
        (20, 10, 79, False),
        (20, 10, 80, True),
        (5, 25, 99, False),
    ],
)
def test_quantile_privacy_uses_stricter_minimum(configured, platform, n, passes):
    transformer = QuantileTransformerApprox(
        n_quantiles=5,
        platform_minimum=platform,
        min_observations_per_interval=configured,
    )
    data = np.arange(float(n)).reshape(-1, 1)
    if passes:
        transformer.fit(data, agg_client=DummyAggClient())
        assert transformer.recipe_.min_observations_per_interval == max(
            configured, platform
        )
    else:
        with pytest.raises(BadInputError, match="global observations"):
            transformer.fit(data, agg_client=DummyAggClient())


def test_quantile_privacy_counts_nonmissing_values_per_feature():
    data = np.column_stack((np.arange(50.0), np.arange(50.0)))
    data[:11, 1] = np.nan
    with pytest.raises(BadInputError, match="global observations"):
        _fit_in_workers(
            lambda: QuantileTransformerApprox(n_quantiles=5, platform_minimum=10),
            [data[:25], data[25:]],
        )


def test_quantile_uses_strictest_platform_minimum_across_sites():
    coordinator = AggregationCoordinator(2)

    def fit_one(worker_id):
        model = QuantileTransformerApprox(
            n_quantiles=5,
            platform_minimum=[10, 20][worker_id],
        )
        model.fit(
            np.arange(40.0).reshape(-1, 1),
            agg_client=SimulatedAggClient(worker_id=worker_id, coordinator=coordinator),
        )
        return model

    with ThreadPoolExecutor(max_workers=2) as executor:
        models = list(executor.map(fit_one, range(2)))
    assert [model.recipe_.min_observations_per_interval for model in models] == [20, 20]
    np.testing.assert_array_equal(models[0].quantiles_, models[1].quantiles_)


def test_quantile_histogram_error_is_bounded_by_bin_width():
    data = np.random.default_rng(5).normal(size=(2000, 1))
    model = QuantileTransformerApprox(
        n_quantiles=20,
        platform_minimum=10,
        n_bins=128,
    ).fit(data, agg_client=DummyAggClient())
    reference = np.quantile(data[:, 0], model.references_, method="inverted_cdf")
    width = np.ptp(data[:, 0]) / model.n_bins
    np.testing.assert_allclose(model.quantiles_[0], reference, atol=width, rtol=0)


def test_quantile_ties_use_midpoint_references():
    recipe = QuantileTransformerRecipe(
        quantiles=[[0, 0, 0, 1, 1]],
        references=[0, 0.25, 0.5, 0.75, 1],
        output_distribution="uniform",
        min_observations_per_interval=10,
        platform_minimum=10,
        n_bins=256,
        max_refinement_rounds=2,
        fit_diagnostics=[
            {
                "histogram_rounds": 0,
                "intervals": [[0, 0], [0, 0], [0, 0], [1, 1], [1, 1]],
                "support_counts": [0, 10, 10, 10, 0],
                "max_interval_width": 0,
                "limited_by_minimum_support": False,
                "limited_by_refinement_budget": False,
            }
        ],
    )
    model = QuantileTransformerApprox.from_recipe(recipe)
    np.testing.assert_array_equal(
        model.transform([0, 1, np.nan]), [0.25, 0.875, np.nan]
    )


def test_quantile_histogram_handles_adjacent_representable_values():
    low = 1e16
    high = np.nextafter(low, np.inf)
    data = np.repeat([low, high], 20)
    model = QuantileTransformerApprox(
        n_quantiles=5,
        platform_minimum=10,
    ).fit(data, agg_client=DummyAggClient())
    assert np.ptp(model.transform([low, high])) > 0


def test_standard_scaler_many_constant_rows_and_sites():
    data = np.tile(np.linspace(0.01, 100, 200), (10000, 1))
    fitted = _fit_in_workers(StandardScaler, np.array_split(data, 10))
    for model in fitted:
        np.testing.assert_array_equal(model.scale_, np.ones(200))
        np.testing.assert_allclose(model.transform(data[:1]), 0, atol=1e-10)


@pytest.mark.parametrize("feature_range", [(0, 1), (-2, 4)])
def test_minmax_constant_feature_custom_range_matches_sklearn(feature_range):
    train = np.array([[3.0], [3.0], [np.nan]])
    test = np.array([[3.0], [9.0], [np.nan]])
    actual = MinMaxScaler(feature_range).fit(train, agg_client=DummyAggClient())
    reference = SklearnMinMaxScaler(feature_range=feature_range).fit(train)
    np.testing.assert_allclose(
        actual.transform(test), reference.transform(test), equal_nan=True
    )


@pytest.mark.parametrize(
    "parts",
    [
        [np.array([[1e300], [2e300]]), np.array([[1.0], [2.0]])],
        [np.array([[1.0], [2.0]]), np.array([[-1e300], [-2e300]])],
    ],
)
def test_yeo_johnson_extreme_values_use_safe_search_bounds(parts):
    models = _fit_in_workers(YeoJohnsonTransformer, parts)
    np.testing.assert_array_equal(models[0].lambdas_, models[1].lambdas_)
    for model, part in zip(models, parts):
        assert np.isfinite(model.transform(part)).all()
        assert model.fit_diagnostics_[0].evaluations <= model.max_evaluations


def test_yeo_johnson_no_safe_interval_fails_on_all_sites():
    parts = [np.array([[1e300], [2e300]]), np.array([[-1e300], [-2e300]])]
    coordinator = AggregationCoordinator(len(parts))

    def fit_one(item):
        worker_id, part = item
        try:
            YeoJohnsonTransformer().fit(
                part,
                agg_client=SimulatedAggClient(
                    worker_id=worker_id, coordinator=coordinator
                ),
            )
        except BadInputError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=len(parts)) as executor:
        futures = [executor.submit(fit_one, item) for item in enumerate(parts)]
        try:
            outcomes = [future.result(timeout=30) for future in futures]
        except Exception as exc:
            coordinator.abort(exc)
            raise
    # Every site must observe the shared numeric failure without relying on
    # test-harness cancellation to wake up another aggregation participant.
    assert all(
        isinstance(result, BadInputError) and "numeric range" in str(result)
        for result in outcomes
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
    model = factory().fit(data, agg_client=DummyAggClient())
    replay = type(model).from_recipe(json.loads(model.recipe_.model_dump_json()))
    np.testing.assert_allclose(
        replay.transform(data), model.transform(data), equal_nan=True
    )


def test_categorical_imputer_json_roundtrip():
    data = np.array([["a"], ["a"], ["b"], [None]], dtype=object)
    model = SimpleImputer("most_frequent", categories=[["a", "b"]]).fit(
        data, agg_client=DummyAggClient()
    )
    replay = SimpleImputer.from_recipe(json.loads(model.recipe_.model_dump_json()))
    assert replay.transform(data).tolist() == [["a"], ["a"], ["b"], ["a"]]


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
        factory().transform([[1.0]])


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
    model = factory().fit(train, agg_client=DummyAggClient())
    data = np.array([0, np.nan, 5])
    copy = data.copy()
    actual = model.transform(data)
    assert actual.shape == data.shape
    assert np.isnan(actual[1])
    np.testing.assert_array_equal(data, copy)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda recipe: recipe.update(scale=[0]),
        lambda recipe: recipe.update(scale=[1, 2]),
        lambda recipe: recipe.update(location=[np.nan]),
    ],
)
def test_scaler_replay_rejects_invalid_fitted_state(mutate):
    model = StandardScaler().fit([[1], [2]], agg_client=DummyAggClient())
    recipe = model.recipe_.model_dump()
    mutate(recipe)
    with pytest.raises(ValidationError):
        StandardScaler.from_recipe(recipe)


def test_mode_uses_metadata_vocabulary_without_observed_value_union():
    class CountsOnlyClient(DummyAggClient):
        def union(self, values):
            pytest.fail("Observed categories must not be shared.")

    model = SimpleImputer("most_frequent", categories=[["a", "b"]]).fit(
        [["a"], ["a"], [None]], agg_client=CountsOnlyClient()
    )
    assert model.transform([[None]]).tolist() == [["a"]]


def test_yeo_johnson_small_budget_returns_labelled_finite_recipe():
    data = np.array([-4.0, -2.0, -1.0, 0.0, 1.0, 3.0, 8.0])
    model = YeoJohnsonTransformer(max_evaluations=1).fit(
        data, agg_client=DummyAggClient()
    )
    diagnostic = model.recipe_.fit_diagnostics[0]
    assert diagnostic.status == "approximate"
    assert diagnostic.reason == "evaluation_budget_exhausted"
    assert diagnostic.evaluations == 1
    assert np.isfinite(model.transform(data)).all()
