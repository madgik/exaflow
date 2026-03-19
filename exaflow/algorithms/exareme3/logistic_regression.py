from typing import List

from pydantic import BaseModel

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.compose.column_transformer import (
    FederatedColumnTransformer,
)
from exaflow.algorithms.federated.linear_model.logistic_regression import (
    FederatedLogisticRegression,
)
from exaflow.algorithms.federated.pipeline import FederatedPipeline
from exaflow.algorithms.federated.preprocessing import FederatedOneHotEncoder


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


class LogisticRegression(Algorithm):
    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="logistic_regression",
            desc="Federated logistic regression for a binary outcome, with one-hot encoding for categorical covariates.",
            label="Logistic Regression",
            enabled=True,
            inputdata=specs.InputDataSpecifications(
                y=specs.InputDataSpecification(
                    label="Dependent variable (binary)",
                    desc="A unique nominal variable. The variable is converted to binary by assigning 1 to the positive class and 0 to all other classes.",
                    types=[specs.InputDataType.TEXT],
                    stattypes=[specs.InputDataStatType.NOMINAL],
                    required=True,
                    multiple=False,
                    enumslen=None,
                ),
                x=specs.InputDataSpecification(
                    label="Covariates (independent)",
                    desc="One or more covariates (numerical or categorical). Categorical variables are one-hot encoded.",
                    types=[
                        specs.InputDataType.INT,
                        specs.InputDataType.REAL,
                        specs.InputDataType.TEXT,
                    ],
                    stattypes=[
                        specs.InputDataStatType.NUMERICAL,
                        specs.InputDataStatType.NOMINAL,
                    ],
                    required=True,
                    multiple=True,
                    enumslen=None,
                ),
                validation=None,
            ),
            parameters={
                "positive_class": specs.ParameterSpecification(
                    label="Positive class (y=1)",
                    desc="Positive class of y. All other classes are considered negative.",
                    types=[specs.ParameterType.TEXT, specs.ParameterType.INT],
                    required=True,
                    multiple=False,
                    default=None,
                    enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.INPUT_VAR_CDE_ENUMS,
                        source=["y"],
                    ),
                    dict_keys_enums=None,
                    dict_values_enums=None,
                    min=None,
                    max=None,
                ),
            },
            type=specs.AlgorithmType.EXAREME3,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def run(self):
        positive_class = self.get_parameter("positive_class")
        y_var = self.inputdata.y[0]
        x_vars = list(self.inputdata.x)

        model_stats = self.run_local_udf(
            func=local_step,
            kw_args={
                "positive_class": positive_class,
                "y_var": y_var,
                "x_vars": x_vars,
                "metadata": self.metadata,
            },
            identical_results=True,
        )
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
def local_step(
    agg_client,
    data,
    positive_class,
    y_var,
    x_vars,
    metadata,
):
    categorical_vars = [var for var in x_vars if metadata[var]["is_categorical"]]
    numerical_vars = [var for var in x_vars if not metadata[var]["is_categorical"]]

    transformer = FederatedColumnTransformer(
        [("cat", FederatedOneHotEncoder(), categorical_vars)],
        remainder="passthrough",
    )
    pipeline = FederatedPipeline(
        [
            ("features", transformer),
            ("model", FederatedLogisticRegression(fit_intercept=True)),
        ]
    )
    positive_class = FederatedLogisticRegression.coerce_positive_class(
        data[y_var], positive_class
    )
    y = data[y_var].eq(positive_class).to_numpy(dtype=float, copy=False)
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

    summary = results.summary()
    summary["feature_names"] = feature_names
    return summary
