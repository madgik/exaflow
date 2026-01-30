from typing import List
from typing import NamedTuple

import numpy as np
from pydantic import BaseModel
from sklearn.model_selection import KFold

from exaflow.algorithms.exareme3.crossvalidation import buffered_kfold_split
from exaflow.algorithms.exareme3.crossvalidation import min_rows_for_cv
from exaflow.algorithms.exareme3.metrics import build_design_matrix
from exaflow.algorithms.exareme3.metrics import collect_categorical_levels_from_df
from exaflow.algorithms.exareme3.metrics import construct_design_labels
from exaflow.algorithms.exareme3.preprocessing import get_dummy_categories
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.ols import FederatedOLS
from exaflow.worker_communication import BadUserInput

ALGORITHM_NAME = "linear_regression_cv"
ALPHA = 0.05


class BasicStats(NamedTuple):
    mean: float
    std: float


class CVLinearRegressionResult(BaseModel):
    dependent_var: str
    indep_vars: List[str]
    n_obs: List[int]
    mean_sq_error: BasicStats
    r_squared: BasicStats
    mean_abs_error: BasicStats
    f_stat: BasicStats


class LinearRegressionCVAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self):
        y_var = self.inputdata.y[0]
        n_splits = self.get_parameter("n_splits")

        # Identify categorical vs numerical predictors
        categorical_vars = [
            var for var in self.inputdata.x if self.metadata[var]["is_categorical"]
        ]
        numerical_vars = [
            var for var in self.inputdata.x if not self.metadata[var]["is_categorical"]
        ]

        dummy_categories = get_dummy_categories(
            run_local_udf_func=self.run_local_udf,
            categorical_vars=categorical_vars,
            collect_udf=linear_collect_categorical_levels_cv,
        )

        indep_var_names = construct_design_labels(
            categorical_vars=categorical_vars,
            dummy_categories=dummy_categories,
            numerical_vars=numerical_vars,
        )
        # Number of predictors excluding intercept
        p = len(indep_var_names) - 1

        # 1) Check per-worker that n_obs >= n_splits
        check_results = self.run_local_udf(
            func=linear_regression_cv_check_local,
            kw_args={
                "y_var": y_var,
                "n_splits": n_splits,
            },
        )
        if not all(res["ok"] for res in check_results):
            raise BadUserInput(
                "Cross validation cannot run because some of the workers "
                "participating in the experiment have a number of observations "
                f"smaller than the number of splits, {n_splits}."
            )

        # 2) Run distributed K-fold CV
        udf_results = self.run_local_udf(
            func=linear_regression_cv_local_step,
            kw_args={
                "y_var": y_var,
                "categorical_vars": categorical_vars,
                "numerical_vars": numerical_vars,
                "dummy_categories": dummy_categories,
                "n_splits": n_splits,
                "p": p,
            },
        )

        # All workers should return identical global metrics; take the first
        metrics = udf_results[0]

        rmse = np.asarray(metrics["rmse"], dtype=float)
        r2 = np.asarray(metrics["r2"], dtype=float)
        mae = np.asarray(metrics["mae"], dtype=float)
        fstats = np.asarray(metrics["f_stat"], dtype=float)
        nobs = [int(v) for v in metrics["n_obs"]]

        result = CVLinearRegressionResult(
            dependent_var=y_var,
            indep_vars=indep_var_names,
            n_obs=nobs,
            mean_sq_error=BasicStats(
                mean=float(rmse.mean()), std=float(rmse.std(ddof=1))
            ),
            r_squared=BasicStats(mean=float(r2.mean()), std=float(r2.std(ddof=1))),
            mean_abs_error=BasicStats(
                mean=float(mae.mean()), std=float(mae.std(ddof=1))
            ),
            f_stat=BasicStats(mean=float(fstats.mean()), std=float(fstats.std(ddof=1))),
        )
        return result


# ---------------------------------------------------------------------------
# Helper UDFs
# ---------------------------------------------------------------------------


@exareme3_udf()
def linear_collect_categorical_levels_cv(data, categorical_vars):
    """
    Thin UDF wrapper used only to collect categorical levels from workers.

    It delegates the core logic to collect_categorical_levels_from_df so that
    linear/logistic regressions share the same behaviour.
    """

    return collect_categorical_levels_from_df(data, categorical_vars)


@exareme3_udf()
def linear_regression_cv_check_local(data, y_var, n_splits):
    """
    Check on each worker whether the number of observations is at least n_splits.
    """

    return min_rows_for_cv(data, y_var, n_splits)


@exareme3_udf(with_aggregation_server=True)
def linear_regression_cv_local_step(
    agg_client,
    data,
    y_var,
    categorical_vars,
    numerical_vars,
    dummy_categories,
    n_splits,
    p,
):
    """
    Run K-fold CV locally on each worker, but use agg_client to:

    - Train a global linear model per fold (aggregated X'X, X'y, n_train).
    - Aggregate residual statistics on the test set.

    Returns identical global metrics from every worker.
    """
    # Ensure n_splits and p are ints
    n_splits = int(n_splits)
    p = int(p)

    # Build design matrix and target vector
    X = build_design_matrix(
        data,
        categorical_vars=categorical_vars,
        dummy_categories=dummy_categories,
        numerical_vars=numerical_vars,
    )
    y = data[y_var].astype(float).to_numpy().reshape(-1, 1)

    n_rows = X.shape[0]
    if n_rows < n_splits:
        # Should have been caught by the check UDF, but be defensive
        return {
            "n_obs": [],
            "rmse": [],
            "r2": [],
            "mae": [],
            "f_stat": [],
        }

    kf = KFold(n_splits=n_splits, shuffle=False)

    n_obs_per_fold = []
    rmse_per_fold = []
    r2_per_fold = []
    mae_per_fold = []
    fstat_per_fold = []

    for X_train, y_train, X_test, y_test in buffered_kfold_split(
        X, y, n_splits=n_splits
    ):

        # --------------------------
        # Training: global OLS model
        # --------------------------
        model = FederatedOLS(agg_client=agg_client)
        model.fit(X_train, y_train)

        n_train = int(model.nobs)
        coeff = np.asarray(model.params, dtype=float).reshape(-1, 1)

        if n_train == 0 or coeff.size == 0:
            n_obs_per_fold.append(0)
            rmse_per_fold.append(0.0)
            r2_per_fold.append(0.0)
            mae_per_fold.append(0.0)
            fstat_per_fold.append(0.0)
            continue

        # --------------------------
        # Evaluation on test set
        # --------------------------
        if X_test.size == 0:
            n_obs_per_fold.append(n_train)
            rmse_per_fold.append(0.0)
            r2_per_fold.append(0.0)
            mae_per_fold.append(0.0)
            fstat_per_fold.append(0.0)
            continue

        y_pred_local = X_test @ coeff  # (n_test_local, 1)
        resid_local = y_test - y_pred_local

        rss_local = float(np.dot(resid_local.reshape(-1), resid_local.reshape(-1)))
        sum_abs_resid_local = float(np.abs(resid_local).sum())
        n_test_local = float(y_test.shape[0])

        sum_y_test_local = float(y_test.sum())
        sum_sq_y_test_local = float((y_test**2).sum())

        # Aggregate test statistics across workers
        rss_arr = agg_client.sum(np.array([rss_local], dtype=float))
        sum_abs_resid_arr = agg_client.sum(np.array([sum_abs_resid_local], dtype=float))
        n_test_arr = agg_client.sum(np.array([n_test_local], dtype=float))
        sum_y_test_arr = agg_client.sum(np.array([sum_y_test_local], dtype=float))
        sum_sq_y_test_arr = agg_client.sum(np.array([sum_sq_y_test_local], dtype=float))

        rss = float(np.asarray(rss_arr, dtype=float).reshape(-1)[0])
        sum_abs_resid = float(np.asarray(sum_abs_resid_arr, dtype=float).reshape(-1)[0])
        n_test = int(np.asarray(n_test_arr, dtype=float).reshape(-1)[0])
        sum_y_test = float(np.asarray(sum_y_test_arr, dtype=float).reshape(-1)[0])
        sum_sq_y_test = float(np.asarray(sum_sq_y_test_arr, dtype=float).reshape(-1)[0])

        # Global TSS on test set
        if n_test > 0:
            y_mean_test = sum_y_test / n_test
            tss = (
                sum_sq_y_test
                - 2.0 * y_mean_test * sum_y_test
                + n_test * (y_mean_test**2)
            )
        else:
            tss = 0.0

        # Degrees of freedom: same as original linear regression:
        # df = n_obs_train - p - 1 (p excludes intercept)
        df_resid = n_train - p - 1

        if df_resid <= 0 or n_test == 0 or rss <= 0.0 or tss <= 0.0 or p <= 0:
            r2_val = 0.0
            rmse_val = 0.0
            mae_val = 0.0
            f_val = 0.0
        else:
            r2_val = 1.0 - (rss / tss)
            rmse_val = float(np.sqrt(rss / n_test))
            mae_val = float(sum_abs_resid / n_test)
            f_val = float((tss - rss) * df_resid / (p * rss))

        n_obs_per_fold.append(n_train)
        rmse_per_fold.append(rmse_val)
        r2_per_fold.append(r2_val)
        mae_per_fold.append(mae_val)
        fstat_per_fold.append(f_val)

    return {
        "n_obs": n_obs_per_fold,
        "rmse": rmse_per_fold,
        "r2": r2_per_fold,
        "mae": mae_per_fold,
        "f_stat": fstat_per_fold,
    }
