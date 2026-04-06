from __future__ import annotations

import threading

import numpy as np
import pytest

from exaflow.algorithms.federated.mixed_effects import FederatedGLMMBinary
from exaflow.algorithms.federated.utils import BadInputError
from tests.standalone_tests.federated_algorithms.mixed_effects.glmm_test_template import (
    CASE_MATRIX,
)
from tests.standalone_tests.federated_algorithms.mixed_effects.glmm_test_template import (
    GLMMCase,
)
from tests.standalone_tests.federated_algorithms.mixed_effects.glmm_test_template import (
    build_binary_model_kwargs,
)
from tests.standalone_tests.federated_algorithms.mixed_effects.glmm_test_template import (
    split_indices_by_center,
)
from tests.standalone_tests.federated_algorithms.mixed_effects.glmm_test_template import (
    synth_glmm_binary_case,
)
from tests.standalone_tests.federated_algorithms.utils import FederatedAlgorithmTest
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    AggregationCoordinator,
)
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    SimulatedAggClient,
)


def _assert_results_equal(left, right, atol=1e-8, rtol=1e-8):
    assert np.allclose(left.theta, right.theta, atol=atol, rtol=rtol)
    assert np.allclose(left.params, right.params, atol=atol, rtol=rtol)
    assert np.isclose(left.sigma_u2, right.sigma_u2, atol=atol, rtol=rtol)
    assert left.nobs == right.nobs
    assert left.n_groups == right.n_groups
    assert left.fit_intercept == right.fit_intercept
    assert left.converged == right.converged


def _assert_behavior(case: GLMMCase, fed, *, X: np.ndarray, y: np.ndarray):
    probs = fed.predict(X)
    assert probs.shape == (X.shape[0],)
    assert np.all(np.isfinite(probs))
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert np.std(probs) > 1e-3

    assert np.isfinite(fed.sigma_u2) and fed.sigma_u2 >= 0.0
    assert fed.nobs == X.shape[0]
    assert fed.n_groups > 0
    assert fed.n_iter <= 50
    assert fed.history is not None
    assert len(fed.history) > 0

    score_vals = np.array([row["score_norm"] for row in fed.history], dtype=float)
    dtheta_vals = np.array([row["dtheta_max"] for row in fed.history], dtype=float)
    assert np.all(np.isfinite(score_vals))
    assert np.all(np.isfinite(dtheta_vals))
    assert score_vals[-1] <= score_vals[0] + 1e-8
    assert dtheta_vals[-1] <= dtheta_vals[0] + 1e-8

    positives = y == 1.0
    negatives = y == 0.0
    if positives.any() and negatives.any():
        assert float(np.mean(probs[positives])) > float(np.mean(probs[negatives]))

    beta_true = np.asarray(case.beta, dtype=float)
    strong_mask = np.abs(beta_true) >= 0.35
    if np.any(strong_mask):
        assert np.array_equal(
            np.sign(fed.params[strong_mask]), np.sign(beta_true[strong_mask])
        )

    if case.name == "low_random_effect_boundary":
        assert fed.sigma_u2 <= 0.25
    if case.name == "high_random_effect_icc":
        assert fed.sigma_u2 >= 0.4
    if case.name == "distribution_stress":
        observed_rate = float(np.mean(y))
        assert observed_rate < 0.25
        assert float(np.mean(probs)) < 0.35
    if case.name == "no_intercept_model":
        assert fed.fit_intercept is False
        assert fed.params.shape[0] == case.n_features


class TestFederatedGLMMBinary(FederatedAlgorithmTest):
    def _split_inputs(self, X, y, n_workers: int):
        x_mat = np.asarray(X["X"], dtype=float)
        center_ids = np.asarray(X["center_ids"])
        y_vec = np.asarray(y, dtype=float)
        idx_parts = split_indices_by_center(center_ids, n_workers=n_workers, seed=123)
        x_parts = [
            {"X": x_mat[idx], "center_ids": center_ids[idx]} for idx in idx_parts
        ]
        y_parts = [y_vec[idx] for idx in idx_parts]
        return x_parts, y_parts, X, y_vec

    def compute_centralized_result(self, X, y, **kwargs):
        coordinator = AggregationCoordinator(n_workers=1)
        agg_client = SimulatedAggClient(worker_id=0, coordinator=coordinator)
        model = FederatedGLMMBinary(**kwargs["model_kwargs"])
        return model.fit(
            np.asarray(X["X"], dtype=float),
            np.asarray(y, dtype=float),
            center_ids=np.asarray(X["center_ids"]),
            agg_client=agg_client,
        )

    def compute_federated_result(self, X, y, *, agg_client, **kwargs):
        model = FederatedGLMMBinary(**kwargs["model_kwargs"])
        return model.fit(
            np.asarray(X["X"], dtype=float),
            np.asarray(y, dtype=float),
            center_ids=np.asarray(X["center_ids"]),
            agg_client=agg_client,
        )

    def compare(self, federated_output, centralized_output, **kwargs):
        case: GLMMCase = kwargs["case"]
        _assert_results_equal(
            federated_output, centralized_output, atol=1e-8, rtol=1e-8
        )
        _assert_behavior(
            case,
            federated_output,
            X=np.asarray(kwargs["X_full"]["X"], dtype=float),
            y=np.asarray(kwargs["y_full"], dtype=float),
        )

    @pytest.mark.parametrize(
        "case", CASE_MATRIX, ids=[case.name for case in CASE_MATRIX]
    )
    def test_federated_algorithm_with_one_worker(self, case):
        X, y, center_ids = synth_glmm_binary_case(case)
        self.run_comparison(
            X={"X": X, "center_ids": center_ids},
            y=y,
            n_workers=1,
            case=case,
            model_kwargs=build_binary_model_kwargs(fit_intercept=case.fit_intercept),
            X_full={"X": X, "center_ids": center_ids},
            y_full=y,
        )

    @pytest.mark.parametrize(
        "case", CASE_MATRIX, ids=[case.name for case in CASE_MATRIX]
    )
    def test_federated_algorithm_with_multiple_workers(self, case):
        X, y, center_ids = synth_glmm_binary_case(case)
        self.run_comparison(
            X={"X": X, "center_ids": center_ids},
            y=y,
            n_workers=3,
            case=case,
            model_kwargs=build_binary_model_kwargs(fit_intercept=case.fit_intercept),
            X_full={"X": X, "center_ids": center_ids},
            y_full=y,
        )


def run_federated_fit(
    X: np.ndarray,
    y: np.ndarray,
    center_ids: np.ndarray,
    *,
    n_workers: int,
    model_kwargs: dict,
):
    coordinator = AggregationCoordinator(n_workers=n_workers)
    parts = split_indices_by_center(center_ids, n_workers=n_workers, seed=123)
    results = [None] * n_workers
    errors = [None] * n_workers

    def _run(worker_id: int):
        try:
            idx = parts[worker_id]
            agg_client = SimulatedAggClient(
                worker_id=worker_id, coordinator=coordinator
            )
            model = FederatedGLMMBinary(**model_kwargs)
            results[worker_id] = model.fit(
                X[idx],
                y[idx],
                center_ids=center_ids[idx],
                agg_client=agg_client,
            )
        except Exception as exc:  # pragma: no cover
            coordinator.abort(exc)
            errors[worker_id] = exc

    threads = [threading.Thread(target=_run, args=(i,)) for i in range(n_workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for err in errors:
        if err is not None:
            raise err

    baseline = results[0]
    for other in results[1:]:
        _assert_results_equal(baseline, other)
    return baseline


def test_glmm_binary_case_matrix_has_expected_coverage():
    assert len(CASE_MATRIX) == 10
    assert {case.name for case in CASE_MATRIX} == {
        "balanced_reference",
        "few_centers_large_n",
        "many_centers_small_n",
        "highly_unbalanced_clusters",
        "low_random_effect_boundary",
        "high_random_effect_icc",
        "near_zero_signal",
        "correlated_predictors",
        "no_intercept_model",
        "distribution_stress",
    }


def test_glmm_binary_result_invariants_and_predict():
    case = CASE_MATRIX[0]
    X, y, center_ids = synth_glmm_binary_case(case)
    fed = run_federated_fit(
        X,
        y,
        center_ids,
        n_workers=3,
        model_kwargs=build_binary_model_kwargs(
            fit_intercept=case.fit_intercept,
            return_history=True,
        ),
    )
    _assert_behavior(case, fed, X=X, y=y)


def test_glmm_binary_laplace_toggle_remains_finite():
    case = CASE_MATRIX[0]
    X, y, center_ids = synth_glmm_binary_case(case)
    coordinator = AggregationCoordinator(n_workers=1)
    agg_client = SimulatedAggClient(worker_id=0, coordinator=coordinator)

    with_laplace = FederatedGLMMBinary(
        **build_binary_model_kwargs(
            fit_intercept=case.fit_intercept,
            add_laplace_corrections=True,
            return_history=True,
        )
    ).fit(X, y, center_ids=center_ids, agg_client=agg_client)

    coordinator = AggregationCoordinator(n_workers=1)
    agg_client = SimulatedAggClient(worker_id=0, coordinator=coordinator)
    without_laplace = FederatedGLMMBinary(
        **build_binary_model_kwargs(
            fit_intercept=case.fit_intercept,
            add_laplace_corrections=False,
            return_history=True,
        )
    ).fit(X, y, center_ids=center_ids, agg_client=agg_client)

    assert np.all(np.isfinite(with_laplace.params))
    assert np.all(np.isfinite(without_laplace.params))
    assert np.isfinite(with_laplace.sigma_u2)
    assert np.isfinite(without_laplace.sigma_u2)
    assert with_laplace.nobs == without_laplace.nobs == X.shape[0]


def test_glmm_binary_invalid_inputs_raise():
    case = CASE_MATRIX[0]
    X, y, center_ids = synth_glmm_binary_case(case)
    y_bad = y.copy()
    y_bad[0] = 2.0

    coordinator = AggregationCoordinator(n_workers=1)
    agg_client = SimulatedAggClient(worker_id=0, coordinator=coordinator)
    model = FederatedGLMMBinary()
    with pytest.raises(BadInputError):
        model.fit(
            X,
            y_bad,
            center_ids=center_ids,
            agg_client=agg_client,
        )
