"""Create an explicitly named missingness indicator without changing source."""

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.derived_numerical_output import (
    DerivedNumericalOutput,
)
from exaflow.algorithms.exareme3.utils.derived_numerical_output import numerical_output
from exaflow.algorithms.exareme3.utils.derived_numerical_output import (
    source_output_parameters,
)
from exaflow.algorithms.exareme3.utils.preprocessing_step import PreprocessingStep
from exaflow.algorithms.federated.feature_engineering import MissingIndicator


class MissingIndicatorStep(DerivedNumericalOutput, PreprocessingStep):
    _allow_categorical_source = True

    @classmethod
    def get_specification(cls):
        return specs.PreprocessingStepSpecification(
            name="missing_indicator",
            label="Missing Indicator",
            desc="Create a numerical flag for missing values while preserving the source.",
            documentation=(
                "Create exactly one caller-named integer column: one for missing values and zero "
                "for observed values. Source values and row order remain unchanged. No aggregation "
                "is required. Apply before imputation if the original missingness is required. "
                "This step has no learned numerical recipe. Methodology uses the missing-value mask "
                "as in scikit-learn MissingIndicator with all selected features retained."
            ),
            enabled=True,
            parameters=source_output_parameters(),
            outputs=numerical_output(specs.InputDataType.INT),
        )

    def _new_transformer(self, *, platform_minimum):
        return MissingIndicator()
