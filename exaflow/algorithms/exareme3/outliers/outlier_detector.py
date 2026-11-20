"""Create a diagnostic outlier flag without changing observations or rows."""

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
from exaflow.algorithms.federated.outliers import OutlierDetector


class OutlierDetectorStep(DerivedNumericalOutput, PreprocessingStep):
    @classmethod
    def get_specification(cls):
        return specs.PreprocessingStepSpecification(
            name="outlier_detector",
            label="Outlier Detector",
            desc="Create an outlier flag using pooled statistics while preserving the source.",
            documentation=(
                "Create a caller-named integer flag without dropping rows or replacing values. "
                "Z-score mode flags absolute population z-scores greater than threshold (default 3). "
                "Robust mode flags distance from the approximate median greater than threshold times "
                "the approximate interquartile range. Degenerate scales use one. Missing values have "
                "flag zero. Aggregated statistics provide common fitted parameters without sharing "
                "raw data. A single-feature recipe replays fitted parameters without fitting. "
                "This step does not return a recipe."
            ),
            enabled=True,
            parameters={
                **source_output_parameters(),
                "mode": choice_parameter(
                    "Detection rule",
                    "Rule used to identify outliers.",
                    ("zscore", "robust"),
                ),
                "threshold": specs.ParameterSpecification(
                    label="Threshold",
                    desc="Controls the distance required to flag an outlier.",
                    types=[specs.ParameterType.REAL],
                    required=False,
                    multiple=False,
                    default=3.0,
                    min=0,
                ),
                "recipe": recipe_parameter(),
            },
            outputs=numerical_output(specs.InputDataType.INT),
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def _new_transformer(self, *, platform_minimum):
        return OutlierDetector(
            mode=self._params.get("mode"), threshold=self._value("threshold")
        )
