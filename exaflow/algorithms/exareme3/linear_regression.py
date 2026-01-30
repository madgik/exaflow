from typing import List

from pydantic import BaseModel

from exaflow.algorithms.exareme3.metrics import build_design_matrix
from exaflow.algorithms.exareme3.metrics import collect_categorical_levels_from_df
from exaflow.algorithms.exareme3.metrics import construct_design_labels
from exaflow.algorithms.exareme3.preprocessing import get_dummy_categories
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.ols import FederatedOLS

ALGORITHM_NAME = "linear_regression"


class LinearRegressionResult(BaseModel):
    dependent_var: str
    n_obs: int
    df_resid: float
    df_model: float
    rse: float
    r_squared: float
    r_squared_adjusted: float
    f_stat: float
    f_pvalue: float
    indep_vars: List[str]
    coefficients: List[float]
    std_err: List[float]
    t_stats: List[float]
    pvalues: List[float]
    lower_ci: List[float]
    upper_ci: List[float]


class LinearRegressionAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self):
        y_var = self.inputdata.y[0]
        categorical_vars = [
            var for var in self.inputdata.x if self.metadata[var]["is_categorical"]
        ]
        numerical_vars = [
            var for var in self.inputdata.x if not self.metadata[var]["is_categorical"]
        ]

        dummy_categories = get_dummy_categories(
            run_local_udf_func=self.run_local_udf,
            categorical_vars=categorical_vars,
            collect_udf=linear_collect_categorical_levels,
        )

        # Construct names of design-matrix columns: Intercept, dummies, numericals
        indep_var_names = construct_design_labels(
            categorical_vars=categorical_vars,
            dummy_categories=dummy_categories,
            numerical_vars=numerical_vars,
        )

        udf_results = self.run_local_udf(
            func=linear_regression_local_step,
            kw_args={
                "y_var": y_var,
                "categorical_vars": categorical_vars,
                "numerical_vars": numerical_vars,
                "dummy_categories": dummy_categories,
            },
        )

        model_stats = udf_results[0]

        return LinearRegressionResult(
            dependent_var=y_var,
            indep_vars=indep_var_names,
            **model_stats,
        )


@exareme3_udf()
def linear_collect_categorical_levels(data, categorical_vars):
    return collect_categorical_levels_from_df(data, categorical_vars)


@exareme3_udf(with_aggregation_server=True)
def linear_regression_local_step(
    agg_client,
    data,
    y_var,
    categorical_vars,
    numerical_vars,
    dummy_categories,
):

    y = data[y_var].to_numpy(dtype=float, copy=False).reshape(-1, 1)
    X = build_design_matrix(
        data,
        categorical_vars=categorical_vars,
        dummy_categories=dummy_categories,
        numerical_vars=numerical_vars,
    )

    model = FederatedOLS(agg_client=agg_client)
    results = model.fit(X, y)
    conf_int = results.conf_int()
    return {
        "n_obs": model.nobs,
        "df_resid": model.df_resid,
        "df_model": model.df_model,
        "rse": model.rse,
        "r_squared": model.rsquared,
        "r_squared_adjusted": model.rsquared_adj,
        "f_stat": model.fvalue,
        "f_pvalue": model.f_pvalue,
        "coefficients": results.params.tolist(),
        "std_err": results.bse.tolist(),
        "t_stats": results.tvalues.tolist(),
        "pvalues": results.pvalues.tolist(),
        "lower_ci": conf_int[:, 0].tolist() if conf_int.size else [],
        "upper_ci": conf_int[:, 1].tolist() if conf_int.size else [],
    }
