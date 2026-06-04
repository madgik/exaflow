from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd
import pytest

from exaflow.algorithms.exareme3.linear_model.cox_regression_stacked import (
    _to_binary_event_array,
)
from exaflow.algorithms.federated.linear_model.cox_regression_stacked import (
    FederatedStackedCoxRegression,
)
from exaflow.algorithms.federated.linear_model.logistic_regression import (
    FederatedLogisticRegression,
)
from exaflow.algorithms.federated.utils import BadInputError
from tests.standalone_tests.federated_algorithms.linear_model._cox_case_utils import (
    CASE_MATRIX,
)
from tests.standalone_tests.federated_algorithms.linear_model._cox_case_utils import (
    CoxCase,
)
from tests.standalone_tests.federated_algorithms.linear_model._cox_case_utils import (
    simulate_cox_case,
)
from tests.standalone_tests.federated_algorithms.linear_model.cox_vs_stat_models import (
    CoxReferencePH,
)
from tests.standalone_tests.federated_algorithms.utils import FederatedAlgorithmTest
from tests.standalone_tests.federated_algorithms.utils.dummy_agg_client import (
    DummyAggClient,
)
from tests.standalone_tests.federated_algorithms.utils.federated_algorithm_test import (
    _simulate_federated_execution,
)


def _fit_centralized_stacking(
    X: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
) -> dict:
    t0 = perf_counter()
    max_time_bins = FederatedStackedCoxRegression._recommended_max_time_bins(
        n_events=int(events.sum()),
        n_covariates=X.shape[1],
    )
    event_times = _manual_distinct_event_times(
        np.unique(times[events > 0]),
        global_max_time=float(np.max(times)),
        max_time_bins=max_time_bins,
    )
    if np.isclose(event_times[0], 0.0):
        stacked_X, stacked_y = FederatedStackedCoxRegression.stack(
            X=X,
            times=times,
            events=events,
            time_bins=event_times,
        )
    else:
        stacked_X, stacked_y = _manual_stack_at_event_times(
            X=X,
            times=times,
            events=events,
            event_times=event_times,
        )
    fit = FederatedLogisticRegression(fit_intercept=False).fit(
        stacked_X,
        stacked_y.astype(float),
        agg_client=DummyAggClient(),
    )
    fit_summary = fit.summary()

    n_covariates = X.shape[1]
    coefficients = np.asarray(fit_summary["coefficients"], dtype=float)[:n_covariates]
    stderr = np.asarray(fit_summary["stderr"], dtype=float)[:n_covariates]
    lower_ci = np.asarray(fit_summary["lower_ci"], dtype=float)[:n_covariates]
    upper_ci = np.asarray(fit_summary["upper_ci"], dtype=float)[:n_covariates]
    hazard_ratios = np.exp(coefficients)
    hr_lower_ci = np.exp(lower_ci)
    hr_upper_ci = np.exp(upper_ci)
    ll = float(fit_summary["ll"])
    ll0 = float(fit_summary["ll0"])
    r_squared_cs = float(fit_summary["r_squared_cs"])

    return {
        "n_obs": int(len(times)),
        "n_events": int(events.sum()),
        "n_stacked_rows": int(len(stacked_y)),
        "n_covariates": int(n_covariates),
        "coefficients": coefficients.tolist(),
        "hazard_ratios": hazard_ratios.tolist(),
        "std_err": stderr.tolist(),
        "lower_ci": lower_ci.tolist(),
        "upper_ci": upper_ci.tolist(),
        "hr_lower_ci": hr_lower_ci.tolist(),
        "hr_upper_ci": hr_upper_ci.tolist(),
        "z_scores": np.asarray(fit_summary["z_scores"], dtype=float)[
            :n_covariates
        ].tolist(),
        "pvalues": np.asarray(fit_summary["pvalues"], dtype=float)[
            :n_covariates
        ].tolist(),
        "df_model": int(fit_summary["df_model"]),
        "df_resid": int(fit_summary["df_resid"]),
        "r_squared_cs": float(r_squared_cs),
        "r_squared_mcf": float(fit_summary["r_squared_mcf"]),
        "ll0": ll0,
        "ll": ll,
        "aic": float(fit_summary["aic"]),
        "bic": float(fit_summary["bic"]),
        "indep_vars": [f"x{i}" for i in range(n_covariates)],
        "time_grid_strategy": "distinct_event_times",
        "n_time_bins_used": (
            int(len(event_times) - 1)
            if np.isclose(np.asarray(event_times, dtype=float).reshape(-1)[0], 0.0)
            else int(len(event_times))
        ),
        "fit_time_sec": float(perf_counter() - t0),
        "method": "survival_stacking",
    }


def _manual_distinct_event_times(
    observed_event_times: np.ndarray,
    *,
    global_max_time: float,
    max_time_bins: int,
) -> np.ndarray:
    event_times = np.unique(np.asarray(observed_event_times, dtype=float).reshape(-1))
    if len(event_times) <= max_time_bins:
        return event_times

    groups = np.array_split(np.arange(len(event_times), dtype=int), max_time_bins)
    boundaries = [0.0]
    for group in groups[:-1]:
        boundaries.append(float(np.nextafter(event_times[group[-1]], np.inf)))
    boundaries.append(float(np.nextafter(global_max_time, np.inf)))
    return np.asarray(boundaries, dtype=float)


def _manual_stack_at_event_times(
    *,
    X: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    event_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(times, kind="mergesort")
    X_sorted = np.asarray(X, dtype=float)[order]
    times_sorted = np.asarray(times, dtype=float)[order]
    events_sorted = np.asarray(events, dtype=float)[order]

    rows = []
    labels = []
    n_time_bins = len(event_times)
    for bin_idx, event_time in enumerate(event_times):
        at_risk = times_sorted >= event_time
        X_risk = X_sorted[at_risk]
        risk_times = times_sorted[at_risk]
        risk_events = events_sorted[at_risk]

        design = np.zeros((len(X_risk), X_sorted.shape[1] + n_time_bins), dtype=float)
        design[:, : X_sorted.shape[1]] = X_risk
        design[:, X_sorted.shape[1] + bin_idx] = 1.0
        rows.append(design)
        labels.append(((risk_times == event_time) & (risk_events > 0)).astype(float))

    return np.vstack(rows), np.concatenate(labels)


class TestFederatedStackedCoxRegression(FederatedAlgorithmTest):
    def _validate_federated_outputs(self, federated_outputs):
        sanitized = []
        for output in federated_outputs:
            cleaned = dict(output)
            cleaned.pop("fit_time_sec", None)
            sanitized.append(cleaned)
        super()._validate_federated_outputs(sanitized)

    def compute_centralized_result(self, X, y, **kwargs):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        times = y[:, 0]
        events = y[:, 1]
        stacking = _fit_centralized_stacking(X, times, events)
        t0 = perf_counter()
        reference = (
            CoxReferencePH(ties="breslow")
            .fit(
                X,
                times,
                events,
                feature_names=stacking["indep_vars"],
            )
            .summary()
        )
        reference["fit_time_sec"] = float(perf_counter() - t0)
        return {
            "stacking": stacking,
            "reference": reference,
        }

    def compute_federated_result(self, X, y, *, agg_client, **kwargs):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        times = y[:, 0]
        events = y[:, 1]
        model = FederatedStackedCoxRegression(time_grid_strategy="distinct_event_times")
        t0 = perf_counter()
        y_survival = np.column_stack([times, events])
        results = model.fit(X, y_survival, agg_client=agg_client)
        summary = results.summary()
        summary["fit_time_sec"] = float(perf_counter() - t0)
        return summary

    def compare(self, federated_output, centralized_output, **kwargs):
        case: CoxCase = kwargs["case"]
        stacking_output = centralized_output["stacking"]
        reference_output = centralized_output["reference"]
        X_full = np.asarray(kwargs["X_full"], dtype=float)

        atol = 1e-6
        fed_coef = np.asarray(federated_output["coefficients"], dtype=float)
        ref_coef = np.asarray(reference_output["coefficients"], dtype=float)
        coef_diff = fed_coef - ref_coef
        fed_hr = np.asarray(federated_output["hazard_ratios"], dtype=float)
        ref_hr = np.asarray(reference_output["hazard_ratios"], dtype=float)
        hr_diff = fed_hr - ref_hr
        fed_stderr = np.asarray(federated_output["std_err"], dtype=float)
        ref_stderr = np.asarray(reference_output["std_err"], dtype=float)
        stderr_diff = fed_stderr - ref_stderr
        fed_pvalues = np.asarray(federated_output["pvalues"], dtype=float)
        ref_pvalues = np.asarray(reference_output["pvalues"], dtype=float)
        pvalue_diff = fed_pvalues - ref_pvalues
        fed_partial_hazard = np.exp(X_full @ fed_coef)
        ref_partial_hazard = np.exp(X_full @ ref_coef)
        corr = np.corrcoef(fed_partial_hazard, ref_partial_hazard)[0, 1]
        max_abs_coef_diff = float(np.max(np.abs(coef_diff)))
        max_abs_hr_diff = float(np.max(np.abs(hr_diff)))
        max_abs_stderr_diff = float(np.max(np.abs(stderr_diff)))
        max_abs_pvalue_diff = float(np.max(np.abs(pvalue_diff)))

        print(
            "\n[cox_regression_stacked standalone]",
            f"case={case.name}",
            f"workers={kwargs.get('n_workers')}",
            f"n_obs={federated_output['n_obs']}",
            f"n_events={federated_output['n_events']}",
            f"stacked_rows={federated_output['n_stacked_rows']}",
            f"time_bins_used={federated_output['n_time_bins_used']}",
        )
        print(
            "  summary:",
            f"max|coef diff|={max_abs_coef_diff:.6f}",
            f"max|hr diff|={max_abs_hr_diff:.6f}",
            f"max|stderr diff|={max_abs_stderr_diff:.6f}",
            f"max|pvalue diff|={max_abs_pvalue_diff:.6f}",
            f"ph_corr={float(corr):.6f}",
        )
        print(
            "  timings:",
            f"fed_stacked={federated_output['fit_time_sec']:.4f}s",
            f"central_stacked={stacking_output['fit_time_sec']:.4f}s",
            f"statsmodels_ref={reference_output['fit_time_sec']:.4f}s",
        )
        print("  fed coefficients:", federated_output["coefficients"])
        print("  ref coefficients:", reference_output["coefficients"])
        print("  coef diff (fed - ref):", coef_diff.tolist())

        # Exact parity with centralized stacking
        assert federated_output["n_obs"] == stacking_output["n_obs"]
        assert federated_output["n_events"] == stacking_output["n_events"]
        assert federated_output["n_stacked_rows"] == stacking_output["n_stacked_rows"]
        assert federated_output["n_covariates"] == stacking_output["n_covariates"]
        assert federated_output["indep_vars"] == stacking_output["indep_vars"]
        assert (
            federated_output["time_grid_strategy"]
            == stacking_output["time_grid_strategy"]
        )
        assert (
            federated_output["n_time_bins_used"] == stacking_output["n_time_bins_used"]
        )
        assert np.allclose(
            federated_output["coefficients"],
            stacking_output["coefficients"],
            atol=atol,
        )
        assert np.allclose(
            federated_output["hazard_ratios"],
            stacking_output["hazard_ratios"],
            atol=atol,
        )
        assert np.allclose(
            federated_output["std_err"],
            stacking_output["std_err"],
            atol=atol,
        )
        assert np.allclose(
            federated_output["lower_ci"],
            stacking_output["lower_ci"],
            atol=atol,
        )
        assert np.allclose(
            federated_output["upper_ci"],
            stacking_output["upper_ci"],
            atol=atol,
        )
        assert np.allclose(
            federated_output["hr_lower_ci"],
            stacking_output["hr_lower_ci"],
            atol=atol,
        )
        assert np.allclose(
            federated_output["hr_upper_ci"],
            stacking_output["hr_upper_ci"],
            atol=atol,
        )
        assert np.allclose(
            federated_output["z_scores"],
            stacking_output["z_scores"],
            atol=atol,
        )
        assert np.allclose(
            federated_output["pvalues"],
            stacking_output["pvalues"],
            atol=atol,
        )
        assert federated_output["df_model"] == stacking_output["df_model"]
        assert federated_output["df_resid"] == stacking_output["df_resid"]
        assert np.isclose(
            federated_output["r_squared_cs"],
            stacking_output["r_squared_cs"],
            atol=atol,
        )
        assert np.isclose(
            federated_output["r_squared_mcf"],
            stacking_output["r_squared_mcf"],
            atol=atol,
        )
        assert np.isclose(federated_output["ll0"], stacking_output["ll0"], atol=atol)
        assert np.isclose(federated_output["ll"], stacking_output["ll"], atol=atol)
        assert np.isclose(federated_output["aic"], stacking_output["aic"], atol=atol)
        assert np.isclose(federated_output["bic"], stacking_output["bic"], atol=atol)

        # Approximate agreement with classical Cox PH
        assert max_abs_coef_diff <= case.coef_atol

        strong_mask = np.abs(np.asarray(case.beta, dtype=float)) >= 0.2
        if np.any(strong_mask):
            assert np.array_equal(
                np.sign(fed_coef[strong_mask]),
                np.sign(ref_coef[strong_mask]),
            )

        assert corr >= case.hazard_corr_min
        assert np.all(np.isfinite(fed_hr))
        assert np.all(np.isfinite(ref_hr))

    @pytest.mark.parametrize(
        "case",
        CASE_MATRIX,
        ids=[case.name for case in CASE_MATRIX],
    )
    def test_federated_algorithm_with_one_worker(self, case):
        X, y = simulate_cox_case(case)
        self.run_comparison(
            X=X,
            y=y,
            n_workers=1,
            case=case,
            X_full=X,
        )

    @pytest.mark.parametrize(
        "case",
        CASE_MATRIX,
        ids=[case.name for case in CASE_MATRIX],
    )
    def test_federated_algorithm_with_multiple_workers(self, case):
        X, y = simulate_cox_case(case)
        self.run_comparison(
            X=X,
            y=y,
            n_workers=3,
            case=case,
            X_full=X,
        )


def test_stacked_cox_case_matrix_has_expected_coverage():
    assert len(CASE_MATRIX) == 10
    case_names = {case.name for case in CASE_MATRIX}
    assert case_names == {
        "balanced_small",
        "balanced_medium",
        "single_feature",
        "heavier_censoring",
        "four_features",
        "mixed_signals",
        "low_signal",
        "one_feature_high_signal",
        "three_features_sparse_events",
        "two_features_large_n",
    }


@pytest.mark.parametrize("positive_class", [2, "2"])
def test_stacked_cox_maps_integer_event_category_to_binary_vector(positive_class):
    event_var = pd.Series([1, 2, 3, 2, 1], dtype="int64")

    events = _to_binary_event_array(
        event_var,
        positive_class=positive_class,
        agg_client=DummyAggClient(),
    )

    np.testing.assert_array_equal(events, np.array([0.0, 1.0, 0.0, 1.0, 0.0]))


def test_distinct_event_time_stacking_skips_pre_event_interval():
    X = np.array([[1.0], [2.0], [3.0]], dtype=float)
    times = np.array([2.0, 4.0, 5.0], dtype=float)
    events = np.array([1.0, 0.0, 1.0], dtype=float)
    event_times = np.array([2.0, 5.0], dtype=float)

    stacked_X, stacked_y = FederatedStackedCoxRegression.stack(
        X=X,
        times=times,
        events=events,
        time_bins=event_times,
    )

    assert stacked_X.shape == (4, 3)
    assert stacked_y.astype(int).tolist() == [1, 0, 0, 1]


def test_distinct_event_time_stacking_accepts_global_times_not_seen_locally():
    X = np.array([[1.0], [2.0], [3.0]], dtype=float)
    times = np.array([2.0, 4.0, 5.0], dtype=float)
    events = np.array([1.0, 0.0, 0.0], dtype=float)
    event_times = np.array([2.0, 3.0], dtype=float)

    stacked_X, stacked_y = FederatedStackedCoxRegression.stack(
        X=X,
        times=times,
        events=events,
        time_bins=event_times,
    )

    assert stacked_X.shape == (5, 3)
    assert stacked_y.astype(int).tolist() == [1, 0, 0, 0, 0]


def test_distinct_event_time_stacking_accepts_global_times_beyond_local_max():
    X = np.array([[1.0], [2.0], [3.0]], dtype=float)
    times = np.array([2.0, 4.0, 5.0], dtype=float)
    events = np.array([1.0, 0.0, 0.0], dtype=float)
    event_times = np.array([2.0, 6.0], dtype=float)

    stacked_X, stacked_y = FederatedStackedCoxRegression.stack(
        X=X,
        times=times,
        events=events,
        time_bins=event_times,
    )

    assert stacked_X.shape == (3, 3)
    assert stacked_y.astype(int).tolist() == [1, 0, 0]


def test_stacked_cox_allows_worker_with_only_censored_rows():
    X_worker0 = np.array([[0.1], [0.4], [-0.2]], dtype=float)
    y_worker0 = np.array([[2.0, 0.0], [5.0, 0.0], [7.0, 0.0]], dtype=float)

    X_worker1 = np.array([[0.7], [-0.3], [1.1], [0.2]], dtype=float)
    y_worker1 = np.array([[3.0, 1.0], [4.0, 1.0], [6.0, 0.0], [8.0, 1.0]], dtype=float)

    outputs = _simulate_federated_execution(
        2,
        lambda worker_id, agg_client: (
            FederatedStackedCoxRegression(time_grid_strategy="distinct_event_times")
            .fit(
                X_worker0 if worker_id == 0 else X_worker1,
                y_worker0 if worker_id == 0 else y_worker1,
                agg_client=agg_client,
                feature_names=["x0"],
            )
            .summary()
        ),
    )

    assert outputs[0]["n_events"] == 3
    assert outputs[1]["n_events"] == 3
    assert outputs[0]["indep_vars"] == ["x0"]
    assert outputs[0]["n_stacked_rows"] > 0


def test_stacked_cox_rejects_when_global_events_are_zero():
    X = np.array([[0.1], [0.4], [-0.2], [0.7]], dtype=float)
    y = np.array([[2.0, 0.0], [5.0, 0.0], [7.0, 0.0], [8.0, 0.0]], dtype=float)

    with pytest.raises(BadInputError, match="observed event"):
        _simulate_federated_execution(
            2,
            lambda worker_id, agg_client: FederatedStackedCoxRegression(
                time_grid_strategy="distinct_event_times"
            ).fit(
                X[:2] if worker_id == 0 else X[2:],
                y[:2] if worker_id == 0 else y[2:],
                agg_client=agg_client,
                feature_names=["x0"],
            ),
        )


def test_stacked_cox_rejects_zero_covariates_after_encoding():
    X = np.empty((4, 0), dtype=float)
    y = np.array([[2.0, 1.0], [5.0, 0.0], [7.0, 1.0], [8.0, 0.0]], dtype=float)

    with pytest.raises(BadInputError, match="at least one covariate"):
        FederatedStackedCoxRegression(time_grid_strategy="distinct_event_times").fit(
            X,
            y,
            agg_client=DummyAggClient(),
            feature_names=[],
        )
