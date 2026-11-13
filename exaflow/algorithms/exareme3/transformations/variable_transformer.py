"""Source-preserving nonlinear transformations using federated mathematics."""

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
from exaflow.algorithms.federated.feature_engineering import AsinhTransformer
from exaflow.algorithms.federated.feature_engineering import Log1pTransformer
from exaflow.algorithms.federated.feature_engineering import QuantileTransformerApprox
from exaflow.algorithms.federated.feature_engineering import SqrtTransformer
from exaflow.algorithms.federated.feature_engineering import YeoJohnsonTransformer
from exaflow.algorithms.federated.utils import BadInputError


def _setting(label, desc, kind, default, minimum):
    return specs.ParameterSpecification(
        label=label,
        desc=desc,
        types=[kind],
        required=False,
        multiple=False,
        default=default,
        min=minimum,
    )


class VariableTransformer(DerivedNumericalOutput, PreprocessingStep):
    @classmethod
    def get_specification(cls):
        return specs.PreprocessingStepSpecification(
            name="variable_transformer",
            label="Variable Transformer",
            desc="Create a nonlinear numerical column while preserving the source.",
            documentation=(
                "Create one named numerical output. log1p and sqrt require nonnegative input; asinh "
                "accepts all real values. Yeo-Johnson fits a global power with budgeted adaptive "
                "profile-likelihood search (64 evaluations, lambda tolerance 1e-5, likelihood tolerance 1e-10). "
                "Quantile methods map approximate global ranks to uniform [0, 1] or standard normal values. "
                "Defaults are 100 reference quantiles, 256 bins and two extra refinement rounds. "
                "The minimum global observations per rank interval is the maximum of the platform minimum "
                "and min_observations_per_interval (default 5). Refinement groups obey that support floor; "
                "exact global extrema are retained. This guard is not differential privacy. "
                "Uncertainty intervals and search termination are recorded in fitted state. Missing values "
                "stay missing. Optional fitted state replays Yeo-Johnson/quantile methods exactly without "
                "refitting, but is not returned through this step's response. Methodology follows NumPy "
                "elementwise functions and scikit-learn power/quantile preprocessing, with explicit approximation."
            ),
            enabled=True,
            parameters={
                **source_output_parameters(),
                "method": choice_parameter(
                    "Transformation",
                    "Function applied to the source.",
                    (
                        "log1p",
                        "sqrt",
                        "asinh",
                        "yeo_johnson",
                        "quantile_uniform",
                        "quantile_normal",
                    ),
                ),
                "recipe": recipe_parameter(),
                "max_evaluations": _setting(
                    "Evaluation budget",
                    "Limits likelihood evaluations per feature.",
                    specs.ParameterType.INT,
                    64,
                    1,
                ),
                "lambda_tolerance": _setting(
                    "Lambda tolerance",
                    "Controls the width of the final search interval.",
                    specs.ParameterType.REAL,
                    1e-5,
                    0,
                ),
                "likelihood_tolerance": _setting(
                    "Likelihood tolerance",
                    "Controls detection of a likelihood plateau.",
                    specs.ParameterType.REAL,
                    1e-10,
                    0,
                ),
                "n_quantiles": _setting(
                    "Reference quantiles",
                    "Controls resolution of the approximate rank map.",
                    specs.ParameterType.INT,
                    100,
                    2,
                ),
                "n_bins": _setting(
                    "Histogram bins",
                    "Controls resolution of each histogram refinement.",
                    specs.ParameterType.INT,
                    256,
                    2,
                ),
                "max_refinement_rounds": _setting(
                    "Refinement budget",
                    "Limits additional histogram refinement rounds.",
                    specs.ParameterType.INT,
                    2,
                    0,
                ),
                "min_observations_per_interval": _setting(
                    "Minimum interval support",
                    "Sets the global support floor together with platform policy.",
                    specs.ParameterType.INT,
                    5,
                    1,
                ),
            },
            outputs=numerical_output(),
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def _new_transformer(self, *, platform_minimum):
        method = self._params.get("method")
        if not isinstance(method, str):
            raise BadInputError("Select a transformation method.")
        if method == "log1p":
            return Log1pTransformer()
        if method == "sqrt":
            return SqrtTransformer()
        if method == "asinh":
            return AsinhTransformer()
        if method == "yeo_johnson":
            return YeoJohnsonTransformer(
                max_evaluations=self._integer_value("max_evaluations"),
                lambda_tolerance=self._value("lambda_tolerance"),
                likelihood_tolerance=self._value("likelihood_tolerance"),
            )
        if method in {"quantile_uniform", "quantile_normal"}:
            return QuantileTransformerApprox(
                n_quantiles=self._integer_value("n_quantiles"),
                output_distribution="uniform"
                if method == "quantile_uniform"
                else "normal",
                min_observations_per_interval=self._integer_value(
                    "min_observations_per_interval"
                ),
                platform_minimum=platform_minimum,
                n_bins=self._integer_value("n_bins"),
                max_refinement_rounds=self._integer_value("max_refinement_rounds"),
            )
        raise BadInputError("Unknown transformation method.")

    def _integer_value(self, name):
        # Integer-valued doubles are the standard worker-side Struct transport.
        value = self._value(name)
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value
