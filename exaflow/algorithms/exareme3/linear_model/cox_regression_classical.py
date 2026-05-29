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
            desc=(
                "Federated classical Cox proportional hazards regression fitted "
                "through partial likelihood with Breslow handling of tied event "
                "times."
            ),
            documentation=(
                "Fit a federated classical Cox proportional hazards regression "
                "model. The algorithm estimates hazard ratios for covariates "
                "while accounting for baseline hazard differences using partial "
                "likelihood with Breslow handling of tied event times."
            ),
            label="Cox Regression Classical",
            enabled=True,
            inputdata=specs.InputDataSpecifications(
                y=specs.InputDataSpecification(
                    label="Time-to-event variable",
                    desc=(
                        "Single positive numerical variable containing follow-up times."
                    ),
                    types=[specs.InputDataType.REAL, specs.InputDataType.INT],
                    stattypes=[specs.InputDataStatType.NUMERICAL],
                    required=True,
                    max_count=1,
                ),
                x=specs.InputDataSpecification(
                    label="Covariates and event indicator",
                    desc=(
                        "One or more covariates plus exactly one event indicator "
                        "variable, referenced by the 'event_var' parameter."
                    ),
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
                ),
                validation=None,
            ),
            parameters={
                "event_var": specs.ParameterSpecification(
                    label="Event indicator variable",
                    desc="Variable from x to use as the binary event indicator.",
                    types=[specs.ParameterType.TEXT],
                    required=True,
                    multiple=False,
                    default=None,
                    enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.INPUT_VAR_NAMES,
                        source=["x"],
                    ),
                    dict_keys_enums=None,
                    dict_values_enums=None,
                    min=None,
                    max=None,
                ),
                "positive_class": specs.ParameterSpecification(
                    label="Positive event class",
                    desc=(
                        "Optional event label treated as event=1 when the event "
                        "indicator is not already encoded as 0/1."
                    ),
                    types=[specs.ParameterType.TEXT, specs.ParameterType.INT],
                    required=False,
                    multiple=False,
                    default=None,
                    enums=None,
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
        time_var = self.inputdata.y[0]
        event_var = self.get_parameter("event_var")
        positive_class = self.get_parameter("positive_class")

        if event_var not in self.inputdata.x:
            raise BadInputError(
                "event_var must refer to one of the selected x variables."
            )

        covariate_vars = [var for var in self.inputdata.x if var != event_var]
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
                "event indicator levels."
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
        "Event indicator must be binary. Provide positive_class when the event "
        "variable is not encoded as 0/1."
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
