from typing import List

from pydantic import BaseModel

from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.compose.column_transformer import (
    FederatedColumnTransformer,
)
from exaflow.algorithms.federated.linear_model.ols import FederatedOLS
from exaflow.algorithms.federated.pipeline import FederatedPipeline
from exaflow.algorithms.federated.preprocessing import FederatedOneHotEncoder
from exaflow.algorithms.specifications import AlgorithmName


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
    ll: float
    aic: float
    bic: float
    indep_vars: List[str]
    coefficients: List[float]
    std_err: List[float]
    t_stats: List[float]
    pvalues: List[float]
    lower_ci: List[float]
    upper_ci: List[float]


class LinearRegression(Algorithm, algname=AlgorithmName.LINEAR_REGRESSION):
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
            ll=model_stats["ll"],
            aic=model_stats["aic"],
            bic=model_stats["bic"],
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
    transformer = FederatedColumnTransformer(
        [("cat", FederatedOneHotEncoder(), categorical_vars)],
        remainder="passthrough",
    )
    pipeline = FederatedPipeline(
        [
            ("features", transformer),
            ("model", FederatedOLS(fit_intercept=True)),
        ]
    )
    y = data[y_var].to_numpy(dtype=float, copy=False)
    results = pipeline.fit(
        agg_client=agg_client,
        data=data,
        y=y,
        categorical_vars=categorical_vars,
        numerical_vars=numerical_vars,
    )
    feature_names = pipeline.get_feature_names_out(
        categorical_vars=categorical_vars,
        numerical_vars=numerical_vars,
    )
    feature_names = ["Intercept"] + feature_names
    conf_int = results.conf_int()
    return {
        "n_obs": results.nobs,
        "df_resid": results.df_resid,
        "df_model": results.df_model,
        "rse": results.rse,
        "r_squared": results.rsquared,
        "r_squared_adjusted": results.rsquared_adj,
        "f_stat": results.fvalue,
        "f_pvalue": results.f_pvalue,
        "ll": results.ll,
        "aic": results.aic,
        "bic": results.bic,
        "coefficients": results.params.tolist(),
        "std_err": results.bse.tolist(),
        "t_stats": results.tvalues.tolist(),
        "pvalues": results.pvalues.tolist(),
        "lower_ci": conf_int[:, 0].tolist() if conf_int.size else [],
        "upper_ci": conf_int[:, 1].tolist() if conf_int.size else [],
        "feature_names": feature_names,
    }
