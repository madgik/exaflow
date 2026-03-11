from __future__ import annotations

import threading

import numpy as np
import pytest

from exaflow.algorithms.federated.mixed_effects import FederatedGLMMBinary
from exaflow.algorithms.federated.utils import BadInputError
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    AggregationCoordinator,
)
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    SimulatedAggClient,
)


def synth_glmm_binary_df(
    *,
    n_centers: int = 12,
    n_features: int = 2,
    n_min: int = 40,
    n_max: int = 80,
    beta: np.ndarray | None = None,
    sigma_u2: float = 0.6,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if beta is None:
        beta = np.array([-0.3] + [0.6] * n_features, dtype=float)

    x_rows = []
    y_rows = []
    c_rows = []
    for j in range(n_centers):
        nj = int(rng.integers(n_min, n_max + 1))
        x_cov = rng.normal(size=(nj, n_features))
        x_design = np.hstack([np.ones((nj, 1), dtype=float), x_cov])
        u_j = rng.normal(loc=0.0, scale=np.sqrt(sigma_u2))
        eta = x_design @ beta + u_j
        p = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(n=1, p=p, size=nj).astype(float)
        x_rows.append(x_cov)
        y_rows.append(y)
        c_rows.append(np.full(nj, f"C{j}", dtype=object))

    X = np.vstack(x_rows).astype(float)
    y = np.concatenate(y_rows).astype(float)
    center_ids = np.concatenate(c_rows)
    return X, y, center_ids


def _split_indices_by_center(
    center_ids: np.ndarray,
    n_workers: int,
    *,
    seed: int = 123,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = np.unique(center_ids)
    rng.shuffle(centers)
    buckets = [centers[i::n_workers] for i in range(n_workers)]
    return [np.where(np.isin(center_ids, b))[0] for b in buckets]


def _assert_results_equal(left, right, atol=1e-8, rtol=1e-8):
    assert np.allclose(left.theta, right.theta, atol=atol, rtol=rtol)
    assert np.allclose(left.params, right.params, atol=atol, rtol=rtol)
    assert np.isclose(left.sigma_u2, right.sigma_u2, atol=atol, rtol=rtol)
    assert left.nobs == right.nobs
    assert left.n_groups == right.n_groups


def _print_clinical_glmm_binary_summary(*, fed) -> None:
    print("\n" + "=" * 92)
    print("GLMM BINARY CLINICAL SUMMARY")
    print("=" * 92)
    print(
        f"Patients (n): {fed.nobs} | Centers: {fed.n_groups} | "
        f"sigma_u2: {fed.sigma_u2:.6f} | converged: {fed.converged} | "
        f"iterations: {fed.n_iter}"
    )
    print(f"Fixed effects (beta): {np.array2string(fed.params, precision=4)}")
    if getattr(fed, "history", None):
        score0 = fed.history[0]["score_norm"]
        scoreN = fed.history[-1]["score_norm"]
        dtheta0 = fed.history[0]["dtheta_max"]
        dthetaN = fed.history[-1]["dtheta_max"]
        print(
            f"Optimization path: score_norm {score0:.6f} -> {scoreN:.6f}, "
            f"dtheta_max {dtheta0:.6f} -> {dthetaN:.6f}"
        )
    print("=" * 92)


def run_federated_fit(
    X: np.ndarray,
    y: np.ndarray,
    center_ids: np.ndarray,
    *,
    n_workers: int,
    model_kwargs: dict,
):
    coordinator = AggregationCoordinator(n_workers=n_workers)
    parts = _split_indices_by_center(center_ids, n_workers=n_workers, seed=123)
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
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for err in errors:
        if err is not None:
            raise err

    baseline = results[0]
    for other in results[1:]:
        _assert_results_equal(baseline, other)
    return baseline


@pytest.mark.parametrize("add_laplace_corrections", [True, False])
def test_glmm_binary_multi_worker_matches_one_worker(add_laplace_corrections):
    X, y, center_ids = synth_glmm_binary_df(seed=42)
    model_kwargs = dict(
        fit_intercept=True,
        max_iters=40,
        ridge=1e-6,
        tol_theta=1e-6,
        tol_score=1e-4,
        add_laplace_corrections=add_laplace_corrections,
        max_step_norm=5.0,
    )
    one_worker = run_federated_fit(
        X, y, center_ids, n_workers=1, model_kwargs=model_kwargs
    )
    three_workers = run_federated_fit(
        X, y, center_ids, n_workers=3, model_kwargs=model_kwargs
    )
    _assert_results_equal(one_worker, three_workers, atol=1e-8, rtol=1e-8)


def test_glmm_binary_result_invariants_and_predict():
    X, y, center_ids = synth_glmm_binary_df(seed=7)
    fed = run_federated_fit(
        X,
        y,
        center_ids,
        n_workers=3,
        model_kwargs=dict(return_history=True),
    )
    _print_clinical_glmm_binary_summary(fed=fed)
    probs = fed.predict(X[:20])
    assert probs.shape == (20,)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert fed.sigma_u2 > 0.0
    assert fed.nobs == X.shape[0]
    assert fed.n_groups == len(np.unique(center_ids))
    assert fed.n_iter <= 50


def test_glmm_binary_invalid_inputs_raise():
    X, y, center_ids = synth_glmm_binary_df(seed=9)
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
