"""Derive clipped values from global quantile bounds, preserving the source."""

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.derived_numerical_output import (
    DerivedNumericalOutput,
)
from exaflow.algorithms.exareme3.utils.derived_numerical_output import numerical_output
from exaflow.algorithms.exareme3.utils.derived_numerical_output import recipe_parameter
from exaflow.algorithms.exareme3.utils.derived_numerical_output import (
    source_output_parameters,
)
from exaflow.algorithms.exareme3.utils.preprocessing_step import PreprocessingStep
from exaflow.algorithms.federated.outliers import Winsorizer


class GlobalWinsorizer(DerivedNumericalOutput, PreprocessingStep):
    @classmethod
    def get_specification(cls):
        return specs.PreprocessingStepSpecification(
            name="global_winsorizer",
            label="Global Winsorizer",
            desc="Create a globally clipped numerical column while preserving the source.",
            documentation=(
                "Clip the derived column between approximate pooled quantiles. Defaults are 0.01 "
                "and 0.99; bounds must satisfy 0 <= lower < upper <= 1. Missing values remain missing. "
                "The source column and all rows are retained. Quantiles use the existing aggregated "
                "percentile computation without sharing raw data. Unlike local outlier winsorization, "
                "all participants use the same fitted bounds. A single-feature recipe replays those "
                "bounds without fitting; this step does not return the recipe."
            ),
            enabled=True,
            parameters={
                **source_output_parameters(),
                "lower_quantile": specs.ParameterSpecification(
                    label="Lower quantile",
                    desc="Quantile defining the lower clipping bound.",
                    types=[specs.ParameterType.REAL],
                    required=False,
                    multiple=False,
                    default=0.01,
                    min=0,
                    max=1,
                ),
                "upper_quantile": specs.ParameterSpecification(
                    label="Upper quantile",
                    desc="Quantile defining the upper clipping bound.",
                    types=[specs.ParameterType.REAL],
                    required=False,
                    multiple=False,
                    default=0.99,
                    min=0,
                    max=1,
                ),
                "recipe": recipe_parameter(),
            },
            outputs=numerical_output(),
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def _new_transformer(self, *, platform_minimum):
        return Winsorizer(self._value("lower_quantile"), self._value("upper_quantile"))
