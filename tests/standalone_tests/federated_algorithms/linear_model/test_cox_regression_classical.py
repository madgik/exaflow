from __future__ import annotations

from time import perf_counter

import numpy as np
import pytest

from exaflow.algorithms.federated.linear_model.cox_regression_classical import (
    FederatedClassicalCoxRegression,
)
from exaflow.algorithms.federated.linear_model.cox_regression_stacked import (
    FederatedStackedCoxRegression,
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


class TestFederatedClassicalCoxRegression(FederatedAlgorithmTest):
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
        feature_names = [f"x{i}" for i in range(X.shape[1])]

        t0 = perf_counter()
        reference = (
            CoxReferencePH(ties="breslow")
            .fit(
                X,
                times,
                events,
                feature_names=feature_names,
            )
            .summary()
        )
        reference["fit_time_sec"] = float(perf_counter() - t0)
        t0 = perf_counter()
        y_survival = np.column_stack([times, events])
        stacked = (
            FederatedStackedCoxRegression(time_grid_strategy="distinct_event_times")
            .fit(
                X,
                y_survival,
                agg_client=DummyAggClient(),
                feature_names=feature_names,
            )
            .summary()
        )
        stacked["fit_time_sec"] = float(perf_counter() - t0)
        return {
            "reference": reference,
            "stacked": stacked,
        }

    def compute_federated_result(self, X, y, *, agg_client, **kwargs):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        times = y[:, 0]
        events = y[:, 1]
        model = FederatedClassicalCoxRegression(ties="breslow")
        t0 = perf_counter()
        y_survival = np.column_stack([times, events])
        results = model.fit(X, y_survival, agg_client=agg_client)
        summary = results.summary()
        summary["fit_time_sec"] = float(perf_counter() - t0)
        return summary

    def compare(self, federated_output, centralized_output, **kwargs):
        case: CoxCase = kwargs["case"]
        reference_output = centralized_output["reference"]
        stacked_output = centralized_output["stacked"]
        X_full = np.asarray(kwargs["X_full"], dtype=float)

        fed_coef = np.asarray(federated_output["coefficients"], dtype=float)
        ref_coef = np.asarray(reference_output["coefficients"], dtype=float)
        stacked_coef = np.asarray(stacked_output["coefficients"], dtype=float)
        coef_diff = fed_coef - ref_coef
        stacked_gap = fed_coef - stacked_coef
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
        ll_diff = float(federated_output["ll"] - reference_output["ll"])

        print(
            "\n[cox_regression_classical standalone]",
            f"case={case.name}",
            f"workers={kwargs.get('n_workers')}",
            f"n_obs={federated_output['n_obs']}",
            f"n_events={federated_output['n_events']}",
            f"event_times={federated_output['n_unique_event_times']}",
        )
        print(
            "  summary:",
            f"max|coef diff|={max_abs_coef_diff:.6f}",
            f"max|hr diff|={max_abs_hr_diff:.6f}",
            f"max|stderr diff|={max_abs_stderr_diff:.6f}",
            f"max|pvalue diff|={max_abs_pvalue_diff:.6f}",
            f"ll diff={ll_diff:.6e}",
            f"ph_corr={float(corr):.6f}",
        )
        print(
            "  timings:",
            f"fed_classical={federated_output['fit_time_sec']:.4f}s",
            f"statsmodels_ref={reference_output['fit_time_sec']:.4f}s",
            f"stacked={stacked_output['fit_time_sec']:.4f}s",
        )
        print("  fed coefficients:", federated_output["coefficients"])
        print("  ref coefficients:", reference_output["coefficients"])
        print("  stacked coefficients:", stacked_output["coefficients"])
        print("  coef diff (fed - ref):", coef_diff.tolist())
        print("  coef diff (fed - stacked):", stacked_gap.tolist())

        assert federated_output["n_obs"] == reference_output["n_obs"]
        assert federated_output["n_events"] == reference_output["n_events"]
        assert federated_output["n_covariates"] == len(reference_output["coefficients"])
        assert federated_output["indep_vars"] == reference_output["indep_vars"]
        assert federated_output["ties"] == reference_output["ties"] == "breslow"
        assert federated_output["df_model"] == reference_output["df_model"]
        assert federated_output["df_resid"] == reference_output["df_resid"]

        assert max_abs_coef_diff <= 5e-2
        assert max_abs_stderr_diff <= 2e-2
        assert max_abs_hr_diff <= 1.5e-1
        assert max_abs_pvalue_diff <= 1.5e-1
        assert abs(ll_diff) <= 5e-2
        assert corr >= 0.995

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


def test_classical_cox_case_matrix_has_expected_coverage():
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


def test_classical_cox_allows_worker_with_only_censored_rows():
    X_worker0 = np.array([[0.1], [0.4], [-0.2]], dtype=float)
    y_worker0 = np.array([[2.0, 0.0], [5.0, 0.0], [7.0, 0.0]], dtype=float)

    X_worker1 = np.array([[0.7], [-0.3], [1.1]], dtype=float)
    y_worker1 = np.array([[3.0, 1.0], [6.0, 0.0], [8.0, 1.0]], dtype=float)

    outputs = _simulate_federated_execution(
        2,
        lambda worker_id, agg_client: (
            FederatedClassicalCoxRegression(ties="breslow")
            .fit(
                X_worker0 if worker_id == 0 else X_worker1,
                y_worker0 if worker_id == 0 else y_worker1,
                agg_client=agg_client,
                feature_names=["x0"],
            )
            .summary()
        ),
    )

    assert outputs[0]["n_events"] == 2
    assert outputs[1]["n_events"] == 2
    assert outputs[0]["indep_vars"] == ["x0"]
    assert np.isfinite(outputs[0]["ll"])


def test_classical_cox_rejects_when_global_events_are_zero():
    X = np.array([[0.1], [0.4], [-0.2], [0.7]], dtype=float)
    y = np.array([[2.0, 0.0], [5.0, 0.0], [7.0, 0.0], [8.0, 0.0]], dtype=float)

    with pytest.raises(BadInputError, match="observed event"):
        _simulate_federated_execution(
            2,
            lambda worker_id, agg_client: FederatedClassicalCoxRegression(
                ties="breslow"
            ).fit(
                X[:2] if worker_id == 0 else X[2:],
                y[:2] if worker_id == 0 else y[2:],
                agg_client=agg_client,
                feature_names=["x0"],
            ),
        )
