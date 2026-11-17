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
from exaflow.algorithms.federated.linear_model.cox_regression_stacked import (
    ALLOWED_TIME_GRID_STRATEGIES,
)
from exaflow.algorithms.federated.linear_model.cox_regression_stacked import (
    FederatedStackedCoxRegression,
)
from exaflow.algorithms.federated.utils import BadInputError
from exaflow.datatypes import CATEGORICAL_CDE_TYPES


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
            desc="Stacked Cox-style regression for time-to-event data.",
            documentation=(
                "Fits a stacked Cox-style model using survival stacking and "
                "logistic regression. Select the "
                "follow-up duration in y, then select the event variable and "
                "covariates in x. The 'event_var' setting identifies which x "
                "variable is used to build the binary event vector; all other "
                "selected x variables are modeled as covariates.\n\n"
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
                "in declared order. The time_grid_strategy parameter controls how event-time "
                "bins are constructed, and n_time_bins is used only for the "
                "uniform time-grid strategy.\n\n"
                "The result includes covariate coefficients, approximate "
                "hazard ratios, Wald statistics, p-values, confidence "
                "intervals, pseudo R-squared values, information criteria, and "
                "the number of time bins used.\n\n"
                "Reference behavior is aligned with discrete-time survival "
                "stacking methodology using logistic regression on expanded "
                "risk-set rows. It is not identical to classical Cox partial "
                "likelihood; use the classical Cox option when exact Cox "
                "partial-likelihood behavior is required."
            ),
            label="Stacked Cox Regression",
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
                desc=("Select the event variable and one or more covariates."),
                types=[
                    specs.InputDataType.REAL,
                    specs.InputDataType.INT,
                    specs.InputDataType.TEXT,
                ],
                stattypes=[
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
                    variable_stattypes=[specs.InputDataStatType.BINARY],
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
                "time_grid_strategy": specs.ParameterSpecification(
                    label="Time grid strategy",
                    desc="Time discretization strategy for survival stacking.",
                    types=[specs.ParameterType.TEXT],
                    required=False,
                    multiple=False,
                    default="distinct_event_times",
                    enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.LIST,
                        source=sorted(ALLOWED_TIME_GRID_STRATEGIES),
                    ),
                ),
                "n_time_bins": specs.ParameterSpecification(
                    label="Uniform time bins",
                    desc=("Bin count used only when time grid strategy is uniform."),
                    types=[specs.ParameterType.INT],
                    required=False,
                    multiple=False,
                    default=10,
                    min=1,
                ),
            },
            type=specs.AlgorithmType.EXAREME3,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def run(self):
        time_var = self.y[0]
        event_var = self.get_parameter("event_var")
        positive_class = self.get_parameter("positive_class")
        time_grid_strategy = self.get_parameter(
            "time_grid_strategy", "distinct_event_times"
        )
        n_time_bins = int(self.get_parameter("n_time_bins", 10))

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
                "cox_regression_stacked requires at least one covariate besides event_var."
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
    metadata,
):
    _validate_time_grid_strategy(time_grid_strategy)

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
