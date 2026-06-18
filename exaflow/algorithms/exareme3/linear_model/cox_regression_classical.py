from __future__ import annotations

from typing import List

import numpy as np
from pydantic import BaseModel

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.compose.column_transformer import (
    FederatedColumnTransformer,
)
from exaflow.algorithms.federated.linear_model.cox_regression_classical import (
    FederatedClassicalCoxRegression,
)
from exaflow.algorithms.federated.linear_model.logistic_regression import (
    FederatedLogisticRegression,
)
from exaflow.algorithms.federated.preprocessing import FederatedOneHotEncoder
from exaflow.algorithms.federated.utils import BadInputError


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
                "The event variable is never one-hot encoded. It is converted "
                "to a single binary event vector where 1 means the selected "
                "event occurred and 0 means censoring or any other category. "
                "Variables stored as 0/1 or false/true are detected "
                "automatically. Use 'positive_class' when the event variable is "
                "categorical, such as diagnosis category or vital status; that "
                "level is converted to event=1 and all other observed levels "
                "are converted to event=0. Because "
                "'event_var' is selected from a multi-variable x input, the "
                "current specification schema cannot expose 'positive_class' "
                "as a dynamic dropdown from the selected event variable.\n\n"
                "Categorical covariates are one-hot encoded before estimation. "
                "The result includes coefficients, hazard ratios, Wald "
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
                types=[specs.InputDataType.REAL, specs.InputDataType.INT],
                stattypes=[specs.InputDataStatType.NUMERICAL],
                required=True,
                max_count=1,
            ),
            x=specs.InputDataSpecification(
                label="Event variable and covariates",
                desc="Select the event variable and one or more covariates.",
                types=[
                    specs.InputDataType.REAL,
                    specs.InputDataType.INT,
                    specs.InputDataType.TEXT,
                ],
                stattypes=[
                    specs.InputDataStatType.NUMERICAL,
                    specs.InputDataStatType.NOMINAL,
                ],
                required=True,
                min_count=2,
            ),
            parameters={
                "event_var": specs.ParameterSpecification(
                    label="Event variable",
                    desc="Variable from x used to build the binary event vector.",
                    types=[specs.ParameterType.TEXT],
                    required=True,
                    multiple=False,
                    enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.INPUT_VAR_NAMES,
                        source=["x"],
                    ),
                ),
                "positive_class": specs.ParameterSpecification(
                    label="Event of interest",
                    desc=(
                        "Event level mapped to 1; other observed levels are "
                        "mapped to 0."
                    ),
                    types=[specs.ParameterType.TEXT, specs.ParameterType.INT],
                    required=False,
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

        covariate_vars = [var for var in self.x if var != event_var]
        if not covariate_vars:
            raise BadInputError(
                "cox_regression_classical requires at least one covariate besides event_var."
            )

        categorical_vars = [
            var for var in covariate_vars if self.metadata[var]["is_categorical"]
        ]
        numerical_vars = [
            var for var in covariate_vars if not self.metadata[var]["is_categorical"]
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


def _to_binary_event_array(series, *, positive_class, agg_client) -> np.ndarray:
    non_null = series.dropna()
    if non_null.empty:
        raise BadInputError("Event indicator contains only missing values.")

    global_levels = list(agg_client.union(non_null.tolist()))
    if positive_class is not None:
        coerced = FederatedLogisticRegression.coerce_positive_class(
            series,
            positive_class,
        )
        coerced_global_levels = {
            FederatedLogisticRegression.coerce_positive_class(series, level)
            for level in global_levels
        }
        if coerced not in coerced_global_levels:
            raise BadInputError(
                "positive_class for event_var should match one of the observed "
                "event variable levels."
            )
        return series.eq(coerced).to_numpy(dtype=float, copy=False)

    try:
        numeric_levels = np.asarray(global_levels, dtype=float)
    except (TypeError, ValueError):
        numeric_levels = None

    if (
        numeric_levels is not None
        and np.isin(np.unique(numeric_levels), [0.0, 1.0]).all()
    ):
        return series.astype(float).to_numpy(copy=False)

    if set(global_levels).issubset({False, True}):
        return series.astype(float).to_numpy(copy=False)

    raise BadInputError(
        "Event variable must be stored as 0/1 or false/true, or positive_class "
        "must be provided to define which observed level maps to event=1."
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
):
    transformer = FederatedColumnTransformer(
        [("cat", FederatedOneHotEncoder(), categorical_vars)],
        remainder="passthrough",
    )
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
    events = _to_binary_event_array(
        data[event_var],
        positive_class=positive_class,
        agg_client=agg_client,
    )

    y_survival = np.column_stack([times.astype(float), events.astype(float)])

    model = FederatedClassicalCoxRegression(ties="breslow")
    results = model.fit(
        X,
        y_survival,
        agg_client=agg_client,
        feature_names=feature_names,
    )
    return results.summary()
