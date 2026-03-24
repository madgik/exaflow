from __future__ import annotations

import threading
import warnings
from dataclasses import dataclass

import numpy as np
import pytest
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from exaflow.algorithms.federated.mixed_effects import FederatedLMM
from exaflow.algorithms.federated.utils import BadInputError
from tests.standalone_tests.federated_algorithms.utils import FederatedAlgorithmTest
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    AggregationCoordinator,
)
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    SimulatedAggClient,
)


@dataclass(frozen=True)
class LMMCase:
    name: str
    beta: tuple[float, ...]
    sigma2: float
    sigma_u2: float
    seed: int
    fit_intercept: bool = True
    n_features: int = 2
    n_centers: int = 12
    n_min: int = 40
    n_max: int = 80
    cluster_sizes: tuple[int, ...] | None = None
    rho: float = 0.0
    residual_dist: str = "normal"
    residual_df: int = 5
    param_atol: float = 5e-3
    param_rtol: float = 5e-3
    sigma_atol: float = 5e-2
    sigma_rtol: float = 1e-1
    icc_atol: float = 5e-2
    boundary_sigma_u2: bool = False


def _build_covariance(n_features: int, rho: float) -> np.ndarray:
    idx = np.arange(n_features)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def synth_lmm_case(case: LMMCase) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(case.seed)
    beta = np.asarray(case.beta, dtype=float)
    expected_beta_len = case.n_features + (1 if case.fit_intercept else 0)
    if beta.shape[0] != expected_beta_len:
        raise ValueError(
            f"beta length {beta.shape[0]} does not match expected {expected_beta_len}."
        )

    if case.cluster_sizes is not None:
        cluster_sizes = np.asarray(case.cluster_sizes, dtype=int)
    else:
        cluster_sizes = rng.integers(case.n_min, case.n_max + 1, size=case.n_centers)

    cov = _build_covariance(case.n_features, case.rho)
    x_rows = []
    y_rows = []
    c_rows = []
    for j, nj in enumerate(cluster_sizes):
        x_cov = rng.multivariate_normal(
            mean=np.zeros(case.n_features),
            cov=cov,
            size=int(nj),
        )
        x_design = x_cov
        if case.fit_intercept:
            x_design = np.hstack([np.ones((int(nj), 1), dtype=float), x_cov])

        u_j = rng.normal(loc=0.0, scale=np.sqrt(case.sigma_u2))
        if case.residual_dist == "normal":
            eps = rng.normal(loc=0.0, scale=np.sqrt(case.sigma2), size=int(nj))
        elif case.residual_dist == "student_t":
            scale = np.sqrt(case.sigma2 * (case.residual_df - 2) / case.residual_df)
            eps = rng.standard_t(df=case.residual_df, size=int(nj)) * scale
        else:
            raise ValueError(f"Unsupported residual_dist: {case.residual_dist}")

        y = x_design @ beta + u_j + eps
        x_rows.append(x_cov.astype(float))
        y_rows.append(y.astype(float))
        c_rows.append(np.full(int(nj), f"C{j}", dtype=object))

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
    return [np.where(np.isin(center_ids, bucket))[0] for bucket in buckets]


def _assert_results_equal(left, right, atol=1e-8, rtol=1e-8):
    assert np.allclose(left.params, right.params, atol=atol, rtol=rtol)
    assert np.allclose(left.bse, right.bse, atol=atol, rtol=rtol)
    assert np.allclose(left.cov_params, right.cov_params, atol=atol, rtol=rtol)
    assert np.isclose(left.sigma2, right.sigma2, atol=atol, rtol=rtol)
    assert np.isclose(left.sigma_u2, right.sigma_u2, atol=atol, rtol=rtol)
    assert np.isclose(left.ll_reml, right.ll_reml, atol=atol, rtol=rtol)
    assert left.nobs == right.nobs
    assert left.n_groups == right.n_groups
    assert left.df_model == right.df_model
    assert left.df_resid == right.df_resid


def _icc(sigma2: float, sigma_u2: float) -> float:
    denom = sigma2 + sigma_u2
    return float(sigma_u2 / denom) if denom > 0.0 else float("nan")


def _build_default_model_kwargs() -> dict[str, float | int | bool]:
    return dict(
        max_iter=120,
        min_iter=10,
        tol=1e-8,
        ridge=1e-8,
        lower_bound=1e-6,
        upper_bound=1e6,
        reg_lambda=1e-1,
        init_sigma2=1.0,
        init_sigma_u2=0.5,
        use_rough_init=True,
        return_history=True,
    )


CASE_MATRIX = [
    LMMCase(
        name="balanced_reference",
        beta=(0.4, 0.8, -0.6),
        sigma2=1.0,
        sigma_u2=0.8,
        seed=11,
        n_centers=18,
        n_min=55,
        n_max=75,
        rho=0.2,
        param_atol=5e-3,
        param_rtol=8e-3,
        sigma_atol=8e-2,
        sigma_rtol=1.5e-1,
    ),
    LMMCase(
        name="few_centers_large_n",
        beta=(0.3, 0.7, -0.5),
        sigma2=1.2,
        sigma_u2=0.9,
        seed=17,
        cluster_sizes=(140, 160, 180, 150, 170),
        rho=0.15,
        param_atol=1.5e-2,
        param_rtol=2e-2,
        sigma_atol=1.2e-1,
        sigma_rtol=2.5e-1,
        icc_atol=7e-2,
    ),
    LMMCase(
        name="many_centers_small_n",
        beta=(0.25, 0.65, -0.45),
        sigma2=0.8,
        sigma_u2=0.5,
        seed=23,
        n_centers=30,
        n_min=12,
        n_max=18,
        rho=0.1,
        param_atol=2e-2,
        param_rtol=4e-2,
        sigma_atol=1.5e-1,
        sigma_rtol=3e-1,
        icc_atol=9e-2,
    ),
    LMMCase(
        name="highly_unbalanced_clusters",
        beta=(0.5, 0.9, -0.7),
        sigma2=1.0,
        sigma_u2=0.7,
        seed=31,
        cluster_sizes=(8, 10, 12, 15, 18, 25, 35, 60, 90, 140, 210, 260),
        rho=0.25,
        param_atol=1.5e-2,
        param_rtol=2e-2,
        sigma_atol=1.5e-1,
        sigma_rtol=2.5e-1,
        icc_atol=8e-2,
    ),
    LMMCase(
        name="low_random_effect_boundary",
        beta=(0.4, 0.8, -0.5),
        sigma2=1.0,
        sigma_u2=0.05,
        seed=43,
        n_centers=20,
        n_min=80,
        n_max=110,
        rho=0.1,
        param_atol=2e-2,
        param_rtol=3e-2,
        sigma_atol=2e-1,
        sigma_rtol=1.0,
        icc_atol=4e-2,
        boundary_sigma_u2=True,
    ),
    LMMCase(
        name="high_random_effect_icc",
        beta=(0.45, 0.7, -0.4),
        sigma2=0.4,
        sigma_u2=2.2,
        seed=59,
        n_centers=18,
        n_min=50,
        n_max=70,
        rho=0.2,
        param_atol=1.5e-2,
        param_rtol=2e-2,
        sigma_atol=2e-1,
        sigma_rtol=2.5e-1,
        icc_atol=7e-2,
    ),
    LMMCase(
        name="near_zero_signal",
        beta=(0.1, 0.2, -0.2),
        sigma2=1.1,
        sigma_u2=0.6,
        seed=67,
        n_centers=24,
        n_min=70,
        n_max=90,
        rho=0.15,
        param_atol=2e-2,
        param_rtol=5e-2,
        sigma_atol=1.5e-1,
        sigma_rtol=2.5e-1,
        icc_atol=8e-2,
    ),
    LMMCase(
        name="correlated_predictors",
        beta=(0.35, 0.95, -0.9),
        sigma2=1.0,
        sigma_u2=0.8,
        seed=71,
        n_centers=22,
        n_min=70,
        n_max=90,
        rho=0.85,
        param_atol=3e-2,
        param_rtol=4e-2,
        sigma_atol=2e-1,
        sigma_rtol=3e-1,
        icc_atol=8e-2,
    ),
    LMMCase(
        name="no_intercept_model",
        beta=(1.1, -0.7),
        sigma2=0.9,
        sigma_u2=0.5,
        seed=83,
        fit_intercept=False,
        n_features=2,
        n_centers=16,
        n_min=60,
        n_max=85,
        rho=0.2,
        param_atol=1e-2,
        param_rtol=2e-2,
        sigma_atol=1.2e-1,
        sigma_rtol=2.0e-1,
        icc_atol=7e-2,
    ),
    LMMCase(
        name="heavy_tailed_noise",
        beta=(0.4, 0.8, -0.6),
        sigma2=1.0,
        sigma_u2=0.9,
        seed=97,
        n_centers=20,
        n_min=80,
        n_max=100,
        rho=0.2,
        residual_dist="student_t",
        residual_df=5,
        param_atol=2e-2,
        param_rtol=4e-2,
        sigma_atol=2.5e-1,
        sigma_rtol=3.5e-1,
        icc_atol=1e-1,
    ),
]


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


def fit_statsmodels_lmm(
    X: np.ndarray,
    y: np.ndarray,
    center_ids: np.ndarray,
    *,
    fit_intercept: bool,
):
    exog = sm.add_constant(X, has_constant="add") if fit_intercept else X
    model = sm.MixedLM(endog=y, exog=exog, groups=center_ids)

    last_exc: Exception | None = None
    candidates = []
    for method in ("powell", "cg", "nm", "lbfgs"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", UserWarning)
                result = model.fit(
                    reml=True,
                    method=method,
                    maxiter=500,
                    disp=False,
                )
            fe_params = np.asarray(result.fe_params, dtype=float)
            sigma2 = float(result.scale)
            sigma_u2 = float(np.asarray(result.cov_re, dtype=float)[0, 0])
            llf = float(result.llf)
            if (
                np.all(np.isfinite(fe_params))
                and np.isfinite(sigma2)
                and np.isfinite(sigma_u2)
                and np.isfinite(llf)
            ):
                candidates.append(result)
        except Exception as exc:  # pragma: no cover - fallback path
            last_exc = exc

    if candidates:
        return max(candidates, key=lambda res: float(res.llf))
    raise RuntimeError("statsmodels MixedLM failed for case") from last_exc


def _assert_case_against_oracle(
    case: LMMCase,
    fed,
    oracle,
    *,
    X: np.ndarray,
):
    beta_sm = np.asarray(oracle.fe_params, dtype=float)
    sigma2_sm = float(oracle.scale)
    sigma_u2_sm = float(np.asarray(oracle.cov_re, dtype=float)[0, 0])

    np.testing.assert_allclose(
        fed.params,
        beta_sm,
        atol=case.param_atol,
        rtol=case.param_rtol,
    )

    if case.boundary_sigma_u2 or sigma_u2_sm < 1e-3:
        assert 0.0 <= fed.sigma_u2 <= max(0.25, 4.0 * max(sigma_u2_sm, case.sigma_u2))
    else:
        assert np.isclose(
            fed.sigma_u2,
            sigma_u2_sm,
            atol=case.sigma_atol,
            rtol=case.sigma_rtol,
        )

    assert np.isclose(
        fed.sigma2,
        sigma2_sm,
        atol=case.sigma_atol,
        rtol=case.sigma_rtol,
    )

    fed_icc = _icc(fed.sigma2, fed.sigma_u2)
    oracle_icc = _icc(sigma2_sm, sigma_u2_sm)
    assert abs(fed_icc - oracle_icc) <= case.icc_atol

    exog = sm.add_constant(X, has_constant="add") if case.fit_intercept else X
    pred_sm = np.asarray(oracle.predict(exog=exog), dtype=float)
    pred_fed = np.asarray(fed.predict(X), dtype=float)
    assert np.corrcoef(pred_fed, pred_sm)[0, 1] >= 0.999

    beta_true = np.asarray(case.beta, dtype=float)
    sign_mask = np.abs(beta_true) >= 0.2
    assert np.array_equal(np.sign(fed.params[sign_mask]), np.sign(beta_true[sign_mask]))

    assert fed.history is not None
    assert len(fed.history) > 0
    ll_vals = np.array([row["ll_reml"] for row in fed.history], dtype=float)
    assert np.all(np.isfinite(ll_vals))
    assert np.all(np.diff(ll_vals) >= -1e-8)
    assert fed.n_iter <= 120
    assert fed.nobs == X.shape[0]


class TestFederatedLMM(FederatedAlgorithmTest):
    def _split_inputs(self, X, y, n_workers: int):
        x_mat = np.asarray(X["X"], dtype=float)
        center_ids = np.asarray(X["center_ids"])
        y_vec = np.asarray(y, dtype=float)
        idx_parts = _split_indices_by_center(center_ids, n_workers=n_workers, seed=123)
        x_parts = [
            {"X": x_mat[idx], "center_ids": center_ids[idx]}
            for idx in idx_parts
        ]
        y_parts = [y_vec[idx] for idx in idx_parts]
        return x_parts, y_parts, X, y_vec

    def compute_centralized_result(self, X, y, **kwargs):
        case: LMMCase = kwargs["case"]
        return fit_statsmodels_lmm(
            np.asarray(X["X"], dtype=float),
            np.asarray(y, dtype=float),
            np.asarray(X["center_ids"]),
            fit_intercept=case.fit_intercept,
        )

    def compute_federated_result(self, X, y, *, agg_client, **kwargs):
        case: LMMCase = kwargs["case"]
        model = FederatedLMM(
            fit_intercept=case.fit_intercept,
            **kwargs["model_kwargs"],
        )
        return model.fit(
            np.asarray(X["X"], dtype=float),
            np.asarray(y, dtype=float),
            center_ids=np.asarray(X["center_ids"]),
            agg_client=agg_client,
        )

    def compare(self, federated_output, centralized_output, **kwargs):
        case: LMMCase = kwargs["case"]
        _assert_case_against_oracle(
            case,
            federated_output,
            centralized_output,
            X=np.asarray(kwargs["X_full"]["X"], dtype=float),
        )

    @pytest.mark.parametrize("case", CASE_MATRIX, ids=[case.name for case in CASE_MATRIX])
    def test_federated_algorithm_with_one_worker(self, case):
        X, y, center_ids = synth_lmm_case(case)
        model_kwargs = _build_default_model_kwargs()
        self.run_comparison(
            X={"X": X, "center_ids": center_ids},
            y=y,
            n_workers=1,
            case=case,
            model_kwargs=model_kwargs,
            X_full={"X": X, "center_ids": center_ids},
        )

    @pytest.mark.parametrize("case", CASE_MATRIX, ids=[case.name for case in CASE_MATRIX])
    def test_federated_algorithm_with_multiple_workers(self, case):
        X, y, center_ids = synth_lmm_case(case)
        model_kwargs = _build_default_model_kwargs()
        self.run_comparison(
            X={"X": X, "center_ids": center_ids},
            y=y,
            n_workers=3,
            case=case,
            model_kwargs=model_kwargs,
            X_full={"X": X, "center_ids": center_ids},
        )


def test_lmm_case_matrix_has_expected_coverage():
    assert len(CASE_MATRIX) == 10
    case_names = {case.name for case in CASE_MATRIX}
    assert case_names == {
        "balanced_reference",
        "few_centers_large_n",
        "many_centers_small_n",
        "highly_unbalanced_clusters",
        "low_random_effect_boundary",
        "high_random_effect_icc",
        "near_zero_signal",
        "correlated_predictors",
        "no_intercept_model",
        "heavy_tailed_noise",
    }


def test_lmm_result_invariants():
    case = CASE_MATRIX[0]
    X, y, center_ids = synth_lmm_case(case)
    fed = run_federated_fit(
        X,
        y,
        center_ids,
        n_workers=3,
        model_kwargs=dict(
            fit_intercept=case.fit_intercept,
            **_build_default_model_kwargs(),
        ),
    )
    p = fed.params.shape[0]
    assert fed.bse.shape[0] == p
    assert fed.pvalues.shape[0] == p
    assert fed.cov_params.shape == (p, p)
    assert fed.sigma2 > 0
    assert fed.sigma_u2 >= 0
    assert fed.n_iter <= 120
    assert fed.nobs == X.shape[0]
    assert fed.n_groups == len(np.unique(center_ids))
    assert fed.fit_intercept is True


def test_lmm_invalid_inputs_raise():
    case = CASE_MATRIX[0]
    X, y, center_ids = synth_lmm_case(case)

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
