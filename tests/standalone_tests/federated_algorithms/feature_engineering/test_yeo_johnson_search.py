"""Partition, replay and numerical edge-case regressions."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import numpy as np
import pytest
from scipy.special import ndtri
from scipy.stats import yeojohnson

from exaflow.algorithms.federated.feature_engineering.yeo_johnson_transformer import (
    YeoJohnsonTransformer,
)
from exaflow.algorithms.federated.feature_engineering.yeo_johnson_transformer import (
    adaptive_lambda_search,
)
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
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    AggregationCoordinator,
)
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    SimulatedAggClient,
)


def _search(objective, bounds=(-40, 40), budget=64, ftol=1e-10):
    return adaptive_lambda_search(
        objective,
        bounds,
        max_evaluations=budget,
        lambda_tolerance=1e-05,
        likelihood_tolerance=ftol,
    )


@pytest.mark.parametrize("target", [-18.0, 18.0])
def test_search_expands_beyond_old_interval(target):
    lambd, diagnostic = _search(lambda x: (x - target) ** 2)
    assert lambd == pytest.approx(target, abs=0.0001)
    assert diagnostic["status"] == "converged"
    assert diagnostic["evaluations"] <= 64


def test_flat_likelihood_stops_early_and_ties_prefer_identity():
    lambd, diagnostic = _search(lambda x: 5.0)
    assert lambd == 1
    assert diagnostic["reason"] == "flat_objective"
    assert diagnostic["status"] == "approximate"
    assert diagnostic["evaluations"] <= 6


@pytest.mark.parametrize("budget", [1, 3, 6, 9])
def test_budget_is_global_across_expansion_and_refinement(budget):
    seen = []

    def objective(x):
        seen.append(x)
        return (x - 18) ** 2

    lambd, diagnostic = _search(objective, budget=budget)
    assert diagnostic["reason"] == "evaluation_budget_exhausted"
    assert diagnostic["status"] == "approximate"
    assert len(seen) == diagnostic["evaluations"] == budget
    assert lambd in seen
    assert (lambd - 18) ** 2 == min(((value - 18) ** 2 for value in seen))


def test_numerical_boundary_keeps_finite_candidate_and_marks_approximation():
    lambd, diagnostic = _search(lambda x: -x, bounds=(-2, 2))
    assert lambd == 2
    assert diagnostic["reason"] == "numerical_bound"
    assert diagnostic["status"] == "approximate"


def test_no_valid_candidate_is_a_clear_error():
    with pytest.raises(BadInputError, match="No valid"):
        _search(lambda x: np.inf, budget=6)


def test_invalid_refinement_is_not_reported_as_converged():
    lambd, diagnostic = _search(lambda x: 0.0 if x == 1 else np.inf)
    assert lambd == 1
    assert diagnostic["status"] == "approximate"


@pytest.mark.parametrize("sign", [-1, 1])
def test_actual_power_transform_fits_outside_old_bounds_and_matches_scipy(sign):
    normal = ndtri((np.arange(201) + 0.5) / 201)
    data = sign * np.expm1(np.log1p(8 * (normal + 4)) / 8)
    expected, expected_lambda = yeojohnson(data)
    assert abs(expected_lambda) > 5
    model = _fit(YeoJohnsonTransformer(), data, agg_client=DummyAggClient())
    assert model.lambdas_[0] == pytest.approx(expected_lambda, abs=0.002)
    np.testing.assert_allclose(
        _transform(model, data), expected, rtol=0.001, atol=0.001
    )
    assert model.fit_diagnostics_[0].evaluations <= 64
    replay = YeoJohnsonTransformer.from_recipe(
        json.loads(json.dumps(asdict(model.recipe_), allow_nan=False))
    )
    np.testing.assert_array_equal(_transform(replay, data), _transform(model, data))
    assert replay.fit_diagnostics_ == model.fit_diagnostics_


@pytest.mark.parametrize("budget", [1, 64])
def test_partitioned_fit_records_identical_diagnostics_and_lambda(budget):
    parts = [np.array([[np.nan, 7], [np.nan, 7]]), np.array([[-3, 1], [0, 2], [4, 3]])]
    coordinator = AggregationCoordinator(len(parts))

    def fit_one(i):
        try:
            return _fit(
                YeoJohnsonTransformer(max_evaluations=budget),
                parts[i],
                agg_client=SimulatedAggClient(worker_id=i, coordinator=coordinator),
            )
        except Exception as exc:
            coordinator.abort(exc)
            raise

    with ThreadPoolExecutor(max_workers=2) as executor:
        models = list(executor.map(fit_one, range(2)))
    assert asdict(models[0].recipe_) == asdict(models[1].recipe_)
    pooled = _fit(
        YeoJohnsonTransformer(max_evaluations=budget),
        np.vstack(parts),
        agg_client=DummyAggClient(),
    )
    np.testing.assert_allclose(models[0].lambdas_, pooled.lambdas_, atol=0.001)


def test_different_search_settings_are_rejected_before_optimizer_rounds():
    coordinator = AggregationCoordinator(2)

    def fit_one(i):
        try:
            _fit(
                YeoJohnsonTransformer(max_evaluations=[1, 64][i]),
                [1, 2, 3],
                agg_client=SimulatedAggClient(worker_id=i, coordinator=coordinator),
            )
        except BadInputError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(fit_one, i) for i in range(2)]
        try:
            outcomes = [future.result(timeout=20) for future in futures]
        except Exception as exc:
            coordinator.abort(exc)
            raise
    assert all(
        (
            isinstance(result, BadInputError) and "settings" in str(result)
            for result in outcomes
        )
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"max_evaluations": 0},
        {"max_evaluations": True},
        {"max_evaluations": 2.5},
        {"lambda_tolerance": 0},
        {"likelihood_tolerance": -1},
        {"likelihood_tolerance": np.nan},
    ],
)
def test_invalid_search_settings_are_bad_input(settings):
    with pytest.raises(BadInputError):
        YeoJohnsonTransformer(**settings)


def test_constant_feature_has_zero_evaluations_and_exact_identity():
    model = _fit(YeoJohnsonTransformer(), [0.1] * 100, agg_client=DummyAggClient())
    assert model.lambdas_.tolist() == [1]
    assert model.fit_diagnostics_[0].status == "constant"
    assert model.fit_diagnostics_[0].evaluations == 0
    np.testing.assert_array_equal(_transform(model, [0.1, np.nan]), [0.1, np.nan])


def test_real_large_offset_plateau_can_stop_without_consuming_budget():
    data = 10000.0 + ndtri((np.arange(201) + 0.5) / 201)
    model = _fit(
        YeoJohnsonTransformer(likelihood_tolerance=0.001),
        data,
        agg_client=DummyAggClient(),
    )
    diagnostic = model.fit_diagnostics_[0]
    assert diagnostic.status == "approximate"
    assert diagnostic.reason == "flat_objective"
    assert diagnostic.evaluations <= 6
    assert model.lambdas_[0] == 1
    assert np.isfinite(_transform(model, data)).all()


@pytest.mark.parametrize("budget", [1, 8, 64])
def test_aggregation_work_is_bounded_by_recorded_evaluations(budget):

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

    client = CountingClient()
    model = _fit(
        YeoJohnsonTransformer(max_evaluations=budget),
        [-4, -2, 0, 1, 8],
        agg_client=client,
    )
    evaluations = model.fit_diagnostics_[0].evaluations
    assert evaluations <= budget
    assert client.calls <= 5 + 2 * evaluations


@pytest.mark.parametrize(
    "mutate",
    [
        lambda recipe: recipe.update(fit_diagnostics=[]),
        lambda recipe: recipe["fit_diagnostics"][0].update(
            status="converged", reason="numerical_bound"
        ),
        lambda recipe: recipe["fit_diagnostics"][0].update(evaluations=65),
        lambda recipe: recipe["fit_diagnostics"][0].update(objective=np.nan),
    ],
)
def test_replay_rejects_inconsistent_search_diagnostics(mutate):
    model = _fit(YeoJohnsonTransformer(), [0, 1, 3], agg_client=DummyAggClient())
    recipe = asdict(model.recipe_)
    mutate(recipe)
    with pytest.raises(BadInputError):
        YeoJohnsonTransformer.from_recipe(recipe)
