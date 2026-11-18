from __future__ import annotations

from typing import List

import numpy as np
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
from exaflow.algorithms.federated.linear_model.cox_regression_classical import (
    FederatedClassicalCoxRegression,
)
from exaflow.algorithms.federated.utils import BadInputError
from exaflow.datatypes import CATEGORICAL_CDE_TYPES


class ClassicalCoxRegressionSummary(BaseModel):
    n_obs: int
    n_events: int
    n_covariates: int
    n_unique_event_times: int
    coefficients: List[float]
    hazard_ratios: List[float]
    std_err: List[float]
    lower_ci: List[float]
    upper_ci: List[float]
    hr_lower_ci: List[float]
    hr_upper_ci: List[float]
    z_scores: List[float]
    pvalues: List[float]
    df_model: int
    df_resid: int
    ll: float
    ties: str
    n_iter: int
    converged: bool
    score_norm: float
    step_norm: float
    method: str


class ClassicalCoxRegressionResult(BaseModel):
    dependent_var: str
    event_var: str
    indep_vars: List[str]
    summary: ClassicalCoxRegressionSummary


class ClassicalCoxRegression(Algorithm):
    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="cox_regression_classical",
            desc="Cox proportional hazards regression for time-to-event data.",
            documentation=(
                "Fits a classical Cox proportional hazards model using partial "
                "likelihood and Breslow handling of tied event times. Select "
                "the follow-up duration in y, then select the "
                "event variable and covariates in x. The 'event_var' setting "
                "identifies which x variable is used to build the event vector; "
                "all other selected x variables are modeled as covariates.\n\n"
                "The event variable must be declared as binary in the data "
                "model; nominal and ordinal variables are rejected and never "
                "collapsed into an event indicator. It is never one-hot "
                "encoded. 'positive_class' is required and must be one of the "
                "level codes declared for the event variable; that level "
                "means the event occurred (event=1) and the other level is "
                "event=0. Nothing is inferred from the stored values. Because "
                "'event_var' is selected from a multi-variable x input, the "
                "current specification schema cannot expose 'positive_class' "
                "as a dynamic dropdown, so the level is validated against the "
                "event variable's declared levels when the analysis runs.\n\n"
                "Nominal covariates are one-hot encoded with the first sorted "
                "level as reference; binary covariates become one 0/1 column in "
                "declared level order and ordinal covariates one integer column "
                "in declared order. The result includes coefficients, hazard ratios, Wald "
                "statistics, p-values, 95% confidence intervals, partial "
                "log-likelihood, and convergence diagnostics.\n\n"
                "Reference behavior is aligned with standard Cox proportional "
                "hazards partial-likelihood methodology, using Breslow tie "
                "handling. Model quantities are computed from aggregated "
                "sufficient statistics without sharing raw data."
            ),
            label="Cox Proportional Hazards Regression",
            enabled=True,
            required_preprocessing=["missing_values_handler"],
            y=specs.InputDataSpecification(
                label="Follow-up time",
                desc="Positive numerical duration until event or censoring.",
                types=[specs.InputDataStatType.NUMERICAL],
                required=True,
                max_count=1,
            ),
            x=specs.InputDataSpecification(
                label="Event variable and covariates",
                desc="Select the event variable and one or more covariates.",
                types=[
                    specs.InputDataStatType.NUMERICAL,
                    *specs.CATEGORICAL_STATTYPES,
                ],
                required=True,
                min_count=2,
            ),
            parameters={
                "event_var": specs.ParameterSpecification(
                    label="Event variable",
                    desc=(
                        "Binary variable from x used to build the event "
                        "vector. Only variables declared as binary are accepted."
                    ),
                    types=[specs.ParameterType.TEXT],
                    required=True,
                    multiple=False,
                    variable_types=[specs.InputDataStatType.BINARY],
                    enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.INPUT_VAR_NAMES,
                        source=["x"],
                    ),
                ),
                "positive_class": specs.ParameterSpecification(
                    label="Event of interest",
                    desc=(
                        "Level code of the binary event variable mapped to 1; "
                        "the other level is mapped to 0. It must be one of the "
                        "levels declared for the event variable."
                    ),
                    types=[specs.ParameterType.TEXT, specs.ParameterType.INT],
                    required=True,
                    multiple=False,
                ),
            },
            type=specs.AlgorithmType.EXAREME3,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def run(self):
        time_var = self.y[0]
        event_var = self.get_parameter("event_var")
        positive_class = self.get_parameter("positive_class")

        if event_var not in self.x:
            raise BadInputError(
                "event_var must refer to one of the selected x variables."
            )
        require_binary_positive_class(
            self.metadata, event_var, positive_class, role="event variable"
        )

        covariate_vars = [var for var in self.x if var != event_var]
        if not covariate_vars:
            raise BadInputError(
                "cox_regression_classical requires at least one covariate besides event_var."
            )

        categorical_vars = [
            var
            for var in covariate_vars
            if self.metadata[var]["type"] in CATEGORICAL_CDE_TYPES
        ]
        numerical_vars = [
            var
            for var in covariate_vars
            if self.metadata[var]["type"] not in CATEGORICAL_CDE_TYPES
        ]

        model_stats = self.run_local_udf(
            func=cox_regression_classical_local_step,
            kw_args={
                "time_var": time_var,
                "event_var": event_var,
                "positive_class": positive_class,
                "categorical_vars": categorical_vars,
                "numerical_vars": numerical_vars,
            },
            identical_results=True,
        )

        return ClassicalCoxRegressionResult(
            dependent_var=time_var,
            event_var=event_var,
            indep_vars=model_stats["indep_vars"],
            summary=ClassicalCoxRegressionSummary(**model_stats),
        )


@exareme3_udf(with_aggregation_server=True)
def cox_regression_classical_local_step(
    agg_client,
    data,
    time_var,
    event_var,
    positive_class,
    categorical_vars,
    numerical_vars,
    metadata,
):
    transformer = build_covariate_transformer(categorical_vars, metadata)
    transformer.fit(
        agg_client=agg_client,
        data=data,
        categorical_vars=categorical_vars,
        numerical_vars=numerical_vars,
    )
    X = transformer.transform(
        data,
        categorical_vars=categorical_vars,
        numerical_vars=numerical_vars,
    )
    feature_names = transformer.get_feature_names_out(
        categorical_vars=categorical_vars,
        numerical_vars=numerical_vars,
    )
    times = data[time_var].to_numpy(dtype=float, copy=False)
    events = encode_positive_class(data[event_var], positive_class)

    y_survival = np.column_stack([times.astype(float), events.astype(float)])

    model = FederatedClassicalCoxRegression(ties="breslow")
    results = model.fit(
        X,
        y_survival,
        agg_client=agg_client,
        feature_names=feature_names,
    )
    return results.summary()
