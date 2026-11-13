"""Derived numerical imputation; the source measurement remains available."""

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
from exaflow.algorithms.federated.feature_engineering import SimpleImputer
from exaflow.algorithms.federated.utils import BadInputError


class GlobalImputer(DerivedNumericalOutput, PreprocessingStep):
    @classmethod
    def get_specification(cls):
        return specs.PreprocessingStepSpecification(
            name="global_imputer",
            label="Global Imputer",
            desc="Create an imputed numerical column while preserving the source.",
            documentation=(
                "Create one numerical column using pooled mean, approximate median, or a constant. "
                "Constant fill defaults to zero. Observed values are unchanged and the source column "
                "is retained with its missing values. Statistics use aggregated quantities without "
                "sharing raw observations. Optional single-feature recipes replay fitted statistics "
                "without refitting; this step does not return a recipe. This numerical specification "
                "does not expose categorical most_frequent; categorical mode in the numerical library "
                "requires an approved metadata vocabulary and exchanges counts only. "
                "Methodology is aligned with scikit-learn SimpleImputer, with approximate median fitting."
            ),
            enabled=True,
            parameters={
                **source_output_parameters(),
                "strategy": choice_parameter(
                    "Imputation strategy",
                    "Statistic or constant used to replace missing values.",
                    ("mean", "median", "constant"),
                ),
                "fill_value": specs.ParameterSpecification(
                    label="Constant replacement",
                    desc="Replacement used by constant imputation.",
                    types=[specs.ParameterType.REAL],
                    required=False,
                    multiple=False,
                    default=0.0,
                ),
                "recipe": recipe_parameter(),
            },
            outputs=numerical_output(),
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def _new_transformer(self, *, platform_minimum):
        strategy = self._params.get("strategy")
        if not isinstance(strategy, str) or strategy not in {
            "mean",
            "median",
            "constant",
        }:
            raise BadInputError("Select a numerical imputation strategy.")
        return SimpleImputer(strategy=strategy, fill_value=self._value("fill_value"))
