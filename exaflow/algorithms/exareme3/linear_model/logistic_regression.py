from typing import List

from pydantic import BaseModel

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.covariate_encoding import (
    build_covariate_transformer,
)
from exaflow.algorithms.exareme3.utils.positive_class import encode_positive_class
from exaflow.algorithms.exareme3.utils.positive_class import (
    require_binary_positive_class,
)
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.linear_model.logistic_regression import (
    FederatedLogisticRegression,
)
from exaflow.algorithms.federated.pipeline import FederatedPipeline
from exaflow.datatypes import CATEGORICAL_CDE_TYPES


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
            desc="Logistic regression for a binary outcome and one or more covariates.",
            documentation=(
                "Fits a logistic regression model for a binary outcome using "
                "numerical and/or categorical covariates. The dependent "
                "variable is converted to binary by assigning 1 to the "
                "selected positive class and 0 to all other classes. "
                "Nominal covariates are one-hot encoded with the first sorted "
                "level as reference; binary covariates become one 0/1 column "
                "in declared level order and ordinal covariates one integer "
                "column in declared order.\n\n"
                "The model includes an intercept and estimates coefficients by "
                "maximizing the binomial log-likelihood with the logistic link "
                "function. The positive class is selected using the "
                "positive_class parameter.\n\n"
                "The result includes coefficients, standard errors, z-scores, "
                "p-values, confidence intervals, degrees of freedom, pseudo "
                "R-squared values, log-likelihoods, AIC, and BIC.\n\n"
                "Reference behavior is aligned with statsmodels logistic "
                "regression using smf.logit(formula, data).fit(method='newton') "
                "for a model with an intercept. The method computes the same "
                "model quantities from aggregated gradient, Hessian, and "
                "log-likelihood statistics without sharing raw data."
            ),
            label="Logistic Regression",
            enabled=True,
            required_preprocessing=["missing_values_handler"],
            y=specs.InputDataSpecification(
                label="Outcome",
                desc=(
                    "Binary outcome. Only variables declared as binary are "
                    "accepted; nominal variables are rejected, never converted."
                ),
                types=[specs.InputDataStatType.BINARY],
                required=True,
                max_count=1,
            ),
            x=specs.InputDataSpecification(
                label="Covariates",
                desc="Numerical or categorical covariates.",
                types=[
                    specs.InputDataStatType.NUMERICAL,
                    *specs.CATEGORICAL_STATTYPES,
                ],
                required=True,
            ),
            parameters={
                "positive_class": specs.ParameterSpecification(
                    label="Positive class",
                    desc="Outcome category treated as the positive outcome.",
                    types=[specs.ParameterType.TEXT, specs.ParameterType.INT],
                    required=True,
                    multiple=False,
                    enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.INPUT_VAR_CDE_ENUMS,
                        source=["y"],
                    ),
                ),
            },
            type=specs.AlgorithmType.EXAREME3,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def run(self):
        positive_class = self.get_parameter("positive_class")
        y_var = self.y[0]
        require_binary_positive_class(
            self.metadata, y_var, positive_class, role="outcome"
        )
        x_vars = list(self.x)
        categorical_vars = [
            var for var in x_vars if self.metadata[var]["type"] in CATEGORICAL_CDE_TYPES
        ]
        numerical_vars = [
            var
            for var in x_vars
            if self.metadata[var]["type"] not in CATEGORICAL_CDE_TYPES
        ]

        model_stats = self.run_local_udf(
            func=local_step,
            kw_args={
                "positive_class": positive_class,
                "y_var": y_var,
                "x_vars": x_vars,
                "categorical_vars": categorical_vars,
                "numerical_vars": numerical_vars,
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
    categorical_vars,
    numerical_vars,
    metadata,
):
    transformer = build_covariate_transformer(categorical_vars, metadata)
    pipeline = FederatedPipeline(
        [
            ("features", transformer),
            ("model", FederatedLogisticRegression(fit_intercept=True)),
        ]
    )
    y = encode_positive_class(data[y_var], positive_class)
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
