"""Numerical regressions retained through the named-column interface."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import numpy as np
import pytest
from scipy.stats import yeojohnson

from exaflow.algorithms.federated.feature_engineering.yeo_johnson_transformer import (
    YeoJohnsonTransformer,
)
from exaflow.algorithms.federated.utils import BadInputError
from tests.standalone_tests.federated_algorithms.feature_engineering.transformation_test_support import (
    fit as _fit,
)
from tests.standalone_tests.federated_algorithms.feature_engineering.transformation_test_support import (
    fit_in_workers as _fit_in_workers,
)
from tests.standalone_tests.federated_algorithms.feature_engineering.transformation_test_support import (
    transform as _transform,
)
from tests.standalone_tests.federated_algorithms.utils.dummy_agg_client import (
    DummyAggClient,
)
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    AggregationCoordinator,
)
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    SimulatedAggClient,
)


def test_yeo_johnson_supports_negative_and_positive_values():
    data = np.array([[-3.0], [-1.0], [0.0], [2.0], [5.0]])
    transformer = _fit(YeoJohnsonTransformer(), data, agg_client=DummyAggClient())
    assert np.all(np.isfinite(_transform(transformer, data)))
    assert transformer.recipe_.lambdas[0] == pytest.approx(transformer.lambdas_[0])
    replay = YeoJohnsonTransformer.from_recipe(
        json.loads(json.dumps(asdict(transformer.recipe_), allow_nan=False))
    )
    assert np.allclose(_transform(replay, data), _transform(transformer, data))


def test_yeo_johnson_is_partition_stable_and_matches_scipy_reference():
    pooled = np.array([[-4.0], [-2.0], [-1.0], [0.0], [1.0], [3.0], [8.0]])
    fitted = _fit_in_workers(
        YeoJohnsonTransformer, [pooled[:2], pooled[2:4], pooled[4:]]
    )
    expected, expected_lambda = yeojohnson(pooled[:, 0])
    assert np.allclose(fitted[0].lambdas_, fitted[1].lambdas_)
    assert np.allclose(fitted[1].lambdas_, fitted[2].lambdas_)
    assert fitted[0].lambdas_[0] == pytest.approx(expected_lambda, abs=0.001)
    actual = np.vstack(
        [
            _transform(fitted[0], pooled[:2]),
            _transform(fitted[1], pooled[2:4]),
            _transform(fitted[2], pooled[4:]),
        ]
    )
    assert np.allclose(actual[:, 0], expected, atol=0.001)


def test_yeo_johnson_uses_global_constant_detection_across_partitions():
    parts = [np.array([[7.0, 1.0], [7.0, 2.0]]), np.array([[1.0, 9.0], [2.0, 9.0]])]
    fitted = _fit_in_workers(YeoJohnsonTransformer, parts)
    assert np.allclose(fitted[0].lambdas_, fitted[1].lambdas_)


def test_yeo_johnson_all_missing_local_partition_still_joins_aggregation():
    fitted = _fit_in_workers(
        YeoJohnsonTransformer,
        [np.array([[np.nan], [np.nan]]), np.array([[1.0], [2.0]])],
    )
    assert np.allclose(fitted[0].lambdas_, fitted[1].lambdas_)


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
        assert np.isfinite(_transform(model, part)).all()
        assert model.fit_diagnostics_[0].evaluations <= model.max_evaluations


def test_yeo_johnson_no_safe_interval_fails_on_all_sites():
    parts = [np.array([[1e300], [2e300]]), np.array([[-1e300], [-2e300]])]
    coordinator = AggregationCoordinator(len(parts))

    def fit_one(item):
        worker_id, part = item
        try:
            _fit(
                YeoJohnsonTransformer(),
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
    assert all(
        (
            isinstance(result, BadInputError) and "numeric range" in str(result)
            for result in outcomes
        )
    )


def test_yeo_johnson_small_budget_returns_labelled_finite_recipe():
    data = np.array([-4.0, -2.0, -1.0, 0.0, 1.0, 3.0, 8.0])
    model = _fit(
        YeoJohnsonTransformer(max_evaluations=1), data, agg_client=DummyAggClient()
    )
    diagnostic = model.recipe_.fit_diagnostics[0]
    assert diagnostic.status == "approximate"
    assert diagnostic.reason == "evaluation_budget_exhausted"
    assert diagnostic.evaluations == 1
    assert np.isfinite(_transform(model, data)).all()
