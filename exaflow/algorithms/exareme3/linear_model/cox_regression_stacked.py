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
from exaflow.algorithms.federated.linear_model.cox_regression_stacked import (
    ALLOWED_TIME_GRID_STRATEGIES,
)
from exaflow.algorithms.federated.linear_model.cox_regression_stacked import (
    FederatedStackedCoxRegression,
)
from exaflow.algorithms.federated.linear_model.logistic_regression import (
    FederatedLogisticRegression,
)
from exaflow.algorithms.federated.preprocessing import FederatedOneHotEncoder
from exaflow.algorithms.federated.utils import BadInputError


class StackedCoxRegressionSummary(BaseModel):
    n_obs: int
    n_events: int
    n_stacked_rows: int
    n_covariates: int
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
    r_squared_cs: float
    r_squared_mcf: float
    ll0: float
    ll: float
    aic: float
    bic: float
    time_grid_strategy: str
    n_time_bins_used: int
    method: str


class StackedCoxRegressionResult(BaseModel):
    dependent_var: str
    event_var: str
    indep_vars: List[str]
    summary: StackedCoxRegressionSummary


class StackedCoxRegression(Algorithm):
    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="cox_regression_stacked",
            desc=(
                "Federated stacked Cox-like regression implemented through survival "
                "stacking and stacked logistic regression. The reported covariate "
                "coefficients approximate Cox proportional-hazards log-hazard "
                "ratios."
            ),
            documentation=(
                "Fit a federated stacked Cox-like regression model. The algorithm "
                "estimates hazard ratios using survival stacking and logistic "
                "regression, approximating Cox proportional-hazards log-hazard ratios."
            ),
            label="Cox Regression Stacked",
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
                "time_grid_strategy": specs.ParameterSpecification(
                    label="Time grid strategy",
                    desc=(
                        "Time discretization strategy. Supported values: "
                        "'distinct_event_times' and 'uniform'."
                    ),
                    types=[specs.ParameterType.TEXT],
                    required=False,
                    multiple=False,
                    default="distinct_event_times",
                    enums=None,
                    dict_keys_enums=None,
                    dict_values_enums=None,
                    min=None,
                    max=None,
                ),
                "n_time_bins": specs.ParameterSpecification(
                    label="Number of time bins",
                    desc=(
                        "Number of uniform time bins when "
                        "time_grid_strategy='uniform'. Ignored for "
                        "distinct_event_times."
                    ),
                    types=[specs.ParameterType.INT],
                    required=False,
                    multiple=False,
                    default=10,
                    enums=None,
                    dict_keys_enums=None,
                    dict_values_enums=None,
                    min=1,
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
        time_grid_strategy = self.get_parameter(
            "time_grid_strategy", "distinct_event_times"
        )
        n_time_bins = int(self.get_parameter("n_time_bins", 10))

        if event_var not in self.inputdata.x:
            raise BadInputError(
                "event_var must refer to one of the selected x variables."
            )

        covariate_vars = [var for var in self.inputdata.x if var != event_var]
        if not covariate_vars:
            raise BadInputError(
                "cox_regression_stacked requires at least one covariate besides event_var."
            )

        categorical_vars = [
            var for var in covariate_vars if self.metadata[var]["is_categorical"]
        ]
        numerical_vars = [
            var for var in covariate_vars if not self.metadata[var]["is_categorical"]
        ]

        model_stats = self.run_local_udf(
            func=cox_regression_stacked_local_step,
            kw_args={
                "time_var": time_var,
                "event_var": event_var,
                "positive_class": positive_class,
                "categorical_vars": categorical_vars,
                "numerical_vars": numerical_vars,
                "time_grid_strategy": time_grid_strategy,
                "n_time_bins": n_time_bins,
            },
            identical_results=True,
        )

        return StackedCoxRegressionResult(
            dependent_var=time_var,
            event_var=event_var,
            indep_vars=model_stats["indep_vars"],
            summary=StackedCoxRegressionSummary(**model_stats),
        )


def _validate_time_grid_strategy(time_grid_strategy: str) -> None:
    if time_grid_strategy not in ALLOWED_TIME_GRID_STRATEGIES:
        raise BadInputError(
            "Unsupported time_grid_strategy. Expected one of "
            f"{sorted(ALLOWED_TIME_GRID_STRATEGIES)}."
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
def cox_regression_stacked_local_step(
    agg_client,
    data,
    time_var,
    event_var,
    positive_class,
    categorical_vars,
    numerical_vars,
    time_grid_strategy,
    n_time_bins,
):
    _validate_time_grid_strategy(time_grid_strategy)

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

    model = FederatedStackedCoxRegression(
        time_grid_strategy=time_grid_strategy,
        n_time_bins=int(n_time_bins),
    )
    results = model.fit(
        X,
        y_survival,
        agg_client=agg_client,
        feature_names=feature_names,
    )
    return results.summary()
