"""Source-preserving scaling using the federated feature-engineering core."""

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.derived_numerical_output import (
    DerivedNumericalOutput,
)
from exaflow.algorithms.exareme3.utils.derived_numerical_output import choice_parameter
from exaflow.algorithms.exareme3.utils.derived_numerical_output import numerical_output
from exaflow.algorithms.exareme3.utils.derived_numerical_output import recipe_parameter
from exaflow.algorithms.exareme3.utils.derived_numerical_output import (
    source_output_parameters,
)
from exaflow.algorithms.exareme3.utils.preprocessing_step import PreprocessingStep
from exaflow.algorithms.federated.feature_engineering import MaxAbsScaler
from exaflow.algorithms.federated.feature_engineering import MinMaxScaler
from exaflow.algorithms.federated.feature_engineering import RobustScalerApprox
from exaflow.algorithms.federated.feature_engineering import StandardScaler
from exaflow.algorithms.federated.utils import BadInputError


class VariableScaler(DerivedNumericalOutput, PreprocessingStep):
    @classmethod
    def get_specification(cls):
        return specs.PreprocessingStepSpecification(
            name="variable_scaler",
            label="Variable Scaler",
            desc="Create a scaled numerical column while preserving the source.",
            documentation=(
                "Create exactly one numerical output with the supplied name. The source remains unchanged. "
                "Standard scaling subtracts the pooled mean and divides by population standard deviation; "
                "robust scaling uses approximate median and interquartile range; min-max scaling uses "
                "the pooled extrema and feature_range (default [0, 1]); max-abs scaling divides by the "
                "pooled maximum absolute value. Constant scales use one. Missing values stay missing. "
                "Computed from aggregated statistics without sharing raw data. An optional single-feature "
                "recipe replays fitted state without refitting. Recipes are not returned by this step. "
                "Methodology is aligned with scikit-learn preprocessing, subject to quantile approximation."
            ),
            enabled=True,
            parameters={
                **source_output_parameters(),
                "method": choice_parameter(
                    "Scaling method",
                    "Method used to scale the source.",
                    ("standard", "robust", "minmax", "maxabs"),
                ),
                "feature_range": specs.ParameterSpecification(
                    label="Min-max range",
                    desc="Target range of min-max scaling.",
                    types=[specs.ParameterType.REAL],
                    required=False,
                    multiple=True,
                    default=[0.0, 1.0],
                ),
                "recipe": recipe_parameter(),
            },
            outputs=numerical_output(),
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def _new_transformer(self, *, platform_minimum):
        method = self._params.get("method")
        if method == "standard":
            return StandardScaler()
        if method == "robust":
            return RobustScalerApprox()
        if method == "minmax":
            return MinMaxScaler(feature_range=self._value("feature_range"))
        if method == "maxabs":
            return MaxAbsScaler()
        raise BadInputError("Unknown scaling method.")
