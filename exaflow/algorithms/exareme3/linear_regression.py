from typing import List

from pydantic import BaseModel

from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.ols import FederatedOLS
from exaflow.algorithms.federated.preprocessing.one_hot_encoder import (
    FederatedOneHotEncoder,
)

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

        udf_results = self.run_local_udf(
            func=linear_regression_local_step,
            kw_args={
                "y_var": y_var,
                "categorical_vars": categorical_vars,
                "numerical_vars": numerical_vars,
            },
        )

        model_stats = udf_results[0]

        return LinearRegressionResult(
            dependent_var=y_var,
            indep_vars=model_stats["feature_names"],
            n_obs=model_stats["n_obs"],
            df_resid=model_stats["df_resid"],
            df_model=model_stats["df_model"],
            rse=model_stats["rse"],
            r_squared=model_stats["r_squared"],
            r_squared_adjusted=model_stats["r_squared_adjusted"],
            f_stat=model_stats["f_stat"],
            f_pvalue=model_stats["f_pvalue"],
            coefficients=model_stats["coefficients"],
            std_err=model_stats["std_err"],
            t_stats=model_stats["t_stats"],
            pvalues=model_stats["pvalues"],
            lower_ci=model_stats["lower_ci"],
            upper_ci=model_stats["upper_ci"],
        )


@exareme3_udf(with_aggregation_server=True)
def linear_regression_local_step(
    agg_client,
    data,
    y_var,
    categorical_vars,
    numerical_vars,
):

    encoder = FederatedOneHotEncoder()
    encoder.fit(
        agg_client=agg_client,
        categorical_vars=categorical_vars,
        numerical_vars=numerical_vars,
        data=data,
    )
    feature_names = encoder.get_feature_names_out(
        categorical_vars=categorical_vars,
        numerical_vars=numerical_vars,
    )
    y = data[y_var].to_numpy(dtype=float, copy=False)
    X = encoder.transform(
        data,
        categorical_vars=categorical_vars,
        numerical_vars=numerical_vars,
    )

    model = FederatedOLS()
    results = model.fit(X, y, agg_client=agg_client)
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
        "feature_names": feature_names,
    }
