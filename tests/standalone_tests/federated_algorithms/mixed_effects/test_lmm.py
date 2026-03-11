from __future__ import annotations

import threading

import numpy as np
import pytest
import statsmodels.api as sm

from exaflow.algorithms.federated.mixed_effects import FederatedLMM
from exaflow.algorithms.federated.mixed_effects.lmm import _LMMRemlAggregator
from exaflow.algorithms.federated.utils import BadInputError
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    AggregationCoordinator,
)
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    SimulatedAggClient,
)


def synth_lmm_df(
    *,
    n_centers: int = 12,
    n_features: int = 2,
    n_min: int = 40,
    n_max: int = 80,
    beta: np.ndarray | None = None,
    sigma2: float = 1.0,
    sigma_u2: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if beta is None:
        beta = np.array([0.3] + [0.5] * n_features, dtype=float)

    x_rows = []
    y_rows = []
    c_rows = []
    for j in range(n_centers):
        nj = int(rng.integers(n_min, n_max + 1))
        x_cov = rng.normal(size=(nj, n_features))
        x_design = np.hstack([np.ones((nj, 1), dtype=float), x_cov])
        u_j = rng.normal(loc=0.0, scale=np.sqrt(sigma_u2))
        eps = rng.normal(loc=0.0, scale=np.sqrt(sigma2), size=nj)
        y = x_design @ beta + u_j + eps
        x_rows.append(x_cov)
        y_rows.append(y)
        c_rows.append(np.full(nj, f"C{j}", dtype=object))

    X = np.vstack(x_rows).astype(float)
    y = np.concatenate(y_rows).astype(float)
    center_ids = np.concatenate(c_rows)
    return X, y, center_ids


def synth_lmm_df_stable(seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    More stable synthetic configuration for variance-component estimation:
    - more centers
    - larger per-center sample sizes
    - stronger between-center variance signal
    """
    return synth_lmm_df(
        n_centers=24,
        n_features=2,
        n_min=60,
        n_max=110,
        sigma2=0.9,
        sigma_u2=1.4,
        seed=seed,
    )


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
    assert np.allclose(left.params, right.params, atol=atol, rtol=rtol)
    assert np.allclose(left.bse, right.bse, atol=atol, rtol=rtol)
    assert np.allclose(left.cov_params, right.cov_params, atol=atol, rtol=rtol)
    assert np.isclose(left.sigma2, right.sigma2, atol=atol, rtol=rtol)
    assert np.isclose(left.sigma_u2, right.sigma_u2, atol=atol, rtol=rtol)
    assert left.nobs == right.nobs
    assert left.n_groups == right.n_groups
    assert left.df_model == right.df_model
    assert left.df_resid == right.df_resid


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
            model = FederatedLMM(**model_kwargs)
            results[worker_id] = model.fit(
                X[idx],
                y[idx],
                center_ids=center_ids[idx],
                agg_client=agg_client,
            )
        except Exception as exc:  # pragma: no cover - thread error capture
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


def fit_statsmodels_lmm(
    X: np.ndarray,
    y: np.ndarray,
    center_ids: np.ndarray,
) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray, np.ndarray, float]:
    exog = sm.add_constant(X, has_constant="add")
    md = sm.MixedLM(endog=y, exog=exog, groups=center_ids)
    res = md.fit(reml=True, method="lbfgs", maxiter=300, disp=False)
    beta_sm = np.asarray(res.fe_params, dtype=float)
    sigma2_sm = float(res.scale)
    sigma_u2_sm = float(np.asarray(res.cov_re, dtype=float)[0, 0])
    bse_sm = np.asarray(res.bse_fe, dtype=float)
    pvalues_sm = np.asarray(res.pvalues[: beta_sm.shape[0]], dtype=float)
    ci_raw = res.conf_int()
    # statsmodels may return either pandas DataFrame or numpy ndarray
    if hasattr(ci_raw, "iloc"):
        ci_sm = np.asarray(ci_raw.iloc[: beta_sm.shape[0], :], dtype=float)
    else:
        ci_sm = np.asarray(ci_raw, dtype=float)[: beta_sm.shape[0], :]
    ll_sm = float(res.llf)
    return beta_sm, sigma2_sm, sigma_u2_sm, bse_sm, pvalues_sm, ci_sm, ll_sm


def _effect_direction(beta: float, pvalue: float) -> str:
    if not np.isfinite(pvalue) or pvalue >= 0.05:
        return "not significant"
    if beta > 0:
        return "higher outcome"
    if beta < 0:
        return "lower outcome"
    return "no change"


def _print_clinical_lmm_summary(
    *,
    feature_names: list[str],
    fed,
    beta_sm: np.ndarray,
    sigma2_sm: float,
    sigma_u2_sm: float,
    pvalues_sm: np.ndarray,
    ci_sm: np.ndarray,
    ll_sm: float,
) -> None:
    def _icc(s2: float, su2: float) -> float:
        den = s2 + su2
        return float(su2 / den) if den > 0 else float("nan")

    print("\n" + "=" * 92)
    print("LMM CLINICAL COMPARISON SUMMARY")
    print("=" * 92)
    print(
        f"Patients (n): {fed.nobs} | Centers: {fed.n_groups} | "
        f"ICC federated: {_icc(fed.sigma2, fed.sigma_u2):.4f} | "
        f"ICC statsmodels: {_icc(sigma2_sm, sigma_u2_sm):.4f}"
    )
    print(
        f"Residual variance sigma2: federated={fed.sigma2:.6f}, "
        f"statsmodels={sigma2_sm:.6f}"
    )
    print(
        f"Center variance sigma_u2: federated={fed.sigma_u2:.6f}, "
        f"statsmodels={sigma_u2_sm:.6f}"
    )
    print(f"REML log-likelihood: federated={fed.ll_reml:.6f}, statsmodels={ll_sm:.6f}")
    if getattr(fed, "history", None):
        ll_path = [float(h.get("ll_reml", np.nan)) for h in fed.history]
        finite_ll = [v for v in ll_path if np.isfinite(v)]
        if finite_ll:
            print(
                f"Federated ll_reml path: start={finite_ll[0]:.6f}, "
                f"end={finite_ll[-1]:.6f}, iters={len(fed.history)}"
            )
    print("-" * 92)
    print(
        f"{'Term':<14} {'Fed beta':>11} {'Fed 95% CI':>24} {'Fed p':>10} "
        f"{'SM beta':>11} {'SM 95% CI':>24} {'SM p':>10} {'Clinical':>14}"
    )
    print("-" * 92)

    for i, name in enumerate(feature_names):
        fed_ci = f"[{fed.conf_int_low[i]:.3f}, {fed.conf_int_high[i]:.3f}]"
        sm_ci = f"[{ci_sm[i, 0]:.3f}, {ci_sm[i, 1]:.3f}]"
        clinical = _effect_direction(float(fed.params[i]), float(fed.pvalues[i]))
        print(
            f"{name:<14} "
            f"{fed.params[i]:>11.4f} {fed_ci:>24} {fed.pvalues[i]:>10.4g} "
            f"{beta_sm[i]:>11.4f} {sm_ci:>24} {pvalues_sm[i]:>10.4g} {clinical:>14}"
        )
    print("=" * 92)


def test_lmm_multi_worker_matches_one_worker():
    X, y, center_ids = synth_lmm_df(seed=5)
    model_kwargs = dict(
        fit_intercept=True,
        max_iter=80,
        tol=1e-8,
        lower_bound=1e-6,
        reg_lambda=1e-1,
        init_sigma2=1.0,
        init_sigma_u2=0.5,
    )

    one_worker = run_federated_fit(
        X, y, center_ids, n_workers=1, model_kwargs=model_kwargs
    )
    three_workers = run_federated_fit(
        X, y, center_ids, n_workers=3, model_kwargs=model_kwargs
    )

    _assert_results_equal(one_worker, three_workers, atol=1e-8, rtol=1e-8)


def test_lmm_against_statsmodels_reml():
    X, y, center_ids = synth_lmm_df_stable(seed=42)
    model_kwargs = dict(
        fit_intercept=True,
        max_iter=80,
        tol=1e-8,
        lower_bound=1e-6,
        reg_lambda=1e-1,
        init_sigma2=1.0,
        init_sigma_u2=0.5,
        return_history=True,
    )
    fed = run_federated_fit(X, y, center_ids, n_workers=1, model_kwargs=model_kwargs)
    beta_sm, sigma2_sm, sigma_u2_sm, _, pvalues_sm, ci_sm, ll_sm = fit_statsmodels_lmm(
        X, y, center_ids
    )

    feature_names = ["Intercept"] + [f"x{i + 1}" for i in range(X.shape[1])]
    _print_clinical_lmm_summary(
        feature_names=feature_names,
        fed=fed,
        beta_sm=beta_sm,
        sigma2_sm=sigma2_sm,
        sigma_u2_sm=sigma_u2_sm,
        pvalues_sm=pvalues_sm,
        ci_sm=ci_sm,
        ll_sm=ll_sm,
    )

    boundary = sigma_u2_sm < 1e-8
    if boundary:
        assert np.allclose(fed.params[1:], beta_sm[1:], rtol=2e-2, atol=2e-3)
        assert np.allclose(fed.params[0], beta_sm[0], rtol=1e-3, atol=3e-2)
    else:
        # In mixed models, intercept/variance terms are sensitive across optimizers.
        # Keep strict agreement on slopes; use looser tolerance for intercept.
        assert np.allclose(fed.params[1:], beta_sm[1:], rtol=3e-3, atol=5e-4)
        assert np.allclose(fed.params[0], beta_sm[0], rtol=2e-2, atol=2e-2)
        assert np.isfinite(fed.sigma2) and fed.sigma2 > 0
        assert np.isfinite(fed.sigma_u2) and fed.sigma_u2 > 0
    assert fed.history is not None and len(fed.history) > 0
    ll_vals = np.array([row["ll_reml"] for row in fed.history], dtype=float)
    assert np.all(np.isfinite(ll_vals))
    # Backtracking step should be monotone non-decreasing in accepted objective.
    assert np.all(np.diff(ll_vals) >= -1e-8)


def test_lmm_result_invariants():
    X, y, center_ids = synth_lmm_df(seed=7)
    fed = run_federated_fit(
        X,
        y,
        center_ids,
        n_workers=3,
        model_kwargs=dict(
            return_history=True,
            max_iter=80,
            tol=1e-8,
            lower_bound=1e-6,
            reg_lambda=1e-1,
        ),
    )
    p = fed.params.shape[0]
    assert fed.bse.shape[0] == p
    assert fed.pvalues.shape[0] == p
    assert fed.cov_params.shape == (p, p)
    assert fed.sigma2 > 0
    assert fed.sigma_u2 > 0
    assert fed.n_iter <= 80
    assert fed.nobs == X.shape[0]
    assert fed.n_groups == len(np.unique(center_ids))


def test_lmm_invalid_inputs_raise():
    X, y, center_ids = synth_lmm_df(seed=9)
    with pytest.raises(BadInputError):
        FederatedLMM(init_sigma2=-1.0)
    with pytest.raises(BadInputError):
        FederatedLMM(init_sigma2=1e-8, lower_bound=1e-6)

    coordinator = AggregationCoordinator(n_workers=1)
    agg_client = SimulatedAggClient(worker_id=0, coordinator=coordinator)
    model = FederatedLMM()
    with pytest.raises(BadInputError):
        model.fit(
            X,
            y,
            center_ids=center_ids[:-1],
            agg_client=agg_client,
        )


def test_lmm_sigma2_not_stuck_at_floor():
    X, y, center_ids = synth_lmm_df_stable(seed=123)
    lower = 1e-6
    fed = run_federated_fit(
        X,
        y,
        center_ids,
        n_workers=1,
        model_kwargs=dict(
            fit_intercept=True,
            max_iter=80,
            tol=1e-8,
            lower_bound=lower,
            reg_lambda=1e-1,
            init_sigma2=1.0,
            init_sigma_u2=0.5,
        ),
    )
    # On this synthetic dataset, sigma2 should not collapse to the bound.
    assert fed.sigma2 > lower * 10.0


def test_lmm_variance_gradient_consistency():
    X, y, center_ids = synth_lmm_df_stable(seed=321)
    coordinator = AggregationCoordinator(n_workers=1)
    agg_client = SimulatedAggClient(worker_id=0, coordinator=coordinator)
    model = FederatedLMM(
        fit_intercept=True,
        lower_bound=1e-6,
        reg_lambda=1e-1,
        init_sigma2=1.0,
        init_sigma_u2=0.5,
    )
    Xw = model._add_intercept(np.asarray(X, dtype=float))
    yw = np.asarray(y, dtype=float)
    beta0 = np.zeros(Xw.shape[1], dtype=float)
    s2, su2 = 1.0, 0.5

    phase_a = model._summaries_node(
        Xw, yw, center_ids, s2, su2, beta0, compute_reml_terms=False
    )
    sxx_packed, sxy, syy, global_hist = model._collect_global_phase_a(
        phase_a, agg_client
    )
    agg = _LMMRemlAggregator(Xw.shape[1])
    agg.accumulate(
        {
            "sxx_packed": sxx_packed,
            "sxy": sxy,
            "syy": syy,
            "global_hist": global_hist,
        }
    )
    beta_gls, a_inv = agg.compute_beta_gls()
    phase_b = model._summaries_node(
        Xw, yw, center_ids, s2, su2, beta_gls, compute_reml_terms=True
    )
    q1, q2, t0, t1, b_sum = model._collect_global_phase_b(phase_b, agg_client)
    agg.accumulate({"q1": q1, "q2": q2, "t0": t0, "t1": t1, "b_packed": b_sum})

    g_analytic = agg.gradient_logscale(beta_gls, s2, su2)
    eps = 1e-6
    phi = np.log(np.array([s2, su2], dtype=float))
    g_num = np.zeros(2, dtype=float)
    for i in range(2):
        d = np.zeros(2, dtype=float)
        d[i] = eps
        sp = np.exp(phi + d)
        sm = np.exp(phi - d)
        fp = agg.objective(beta_gls, float(sp[0]), float(sp[1]))
        fm = agg.objective(beta_gls, float(sm[0]), float(sm[1]))
        g_num[i] = (fp - fm) / (2.0 * eps)

    assert np.allclose(g_analytic, g_num, rtol=1e-3, atol=1e-4)
