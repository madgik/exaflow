from typing import List

from pydantic import BaseModel

from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.logistic_regression import FederatedLogisticRegression
from exaflow.algorithms.federated.preprocessing.one_hot_encoder import (
    FederatedOneHotEncoder,
)

ALGORITHM_NAME = "logistic_regression"


class LogisticRegressionSummary(BaseModel):
    n_obs: int
    coefficients: List[float]
    stderr: List[float]
    lower_ci: List[float]
    upper_ci: List[float]
    z_scores: List[float]
    pvalues: List[float]
    df_model: int
    df_resid: int
    r_squared_cs: float
    r_squared_mcf: float
    ll0: float
    ll: float
    aic: float
    bic: float


class LogisticRegressionResult(BaseModel):
    dependent_var: str
    indep_vars: List[str]
    summary: LogisticRegressionSummary


class LogisticRegressionAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self):
        positive_class = self.get_parameter("positive_class")
        y_var = self.inputdata.y[0]
        categorical_vars = [
            var for var in self.inputdata.x if self.metadata[var]["is_categorical"]
        ]
        numerical_vars = [
            var for var in self.inputdata.x if not self.metadata[var]["is_categorical"]
        ]

        udf_results = self.run_local_udf(
            func=logistic_regression_local_step,
            kw_args={
                "positive_class": positive_class,
                "y_var": y_var,
                "categorical_vars": categorical_vars,
                "numerical_vars": numerical_vars,
            },
        )

        model_stats = udf_results[0]
        summary = LogisticRegressionSummary(
            n_obs=model_stats["n_obs"],
            coefficients=model_stats["coefficients"],
            stderr=model_stats["stderr"],
            lower_ci=model_stats["lower_ci"],
            upper_ci=model_stats["upper_ci"],
            z_scores=model_stats["z_scores"],
            pvalues=model_stats["pvalues"],
            df_model=model_stats["df_model"],
            df_resid=model_stats["df_resid"],
            r_squared_cs=model_stats["r_squared_cs"],
            r_squared_mcf=model_stats["r_squared_mcf"],
            ll0=model_stats["ll0"],
            ll=model_stats["ll"],
            aic=model_stats["aic"],
            bic=model_stats["bic"],
        )

        return LogisticRegressionResult(
            dependent_var=y_var,
            indep_vars=model_stats["feature_names"],
            summary=summary,
        )


@exareme3_udf(with_aggregation_server=True)
def logistic_regression_local_step(
    agg_client,
    data,
    positive_class,
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
    positive_class = FederatedLogisticRegression.coerce_positive_class(
        data[y_var], positive_class
    )
    y = data[y_var].eq(positive_class).to_numpy(dtype=float, copy=False)
    X = encoder.transform(
        data,
        categorical_vars=categorical_vars,
        numerical_vars=numerical_vars,
    )

    model = FederatedLogisticRegression()
    results = model.fit(X, y, agg_client=agg_client)

    summary = results.summary()
    summary["feature_names"] = feature_names
    return summary
