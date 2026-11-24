"""Shared mechanics for source-preserving numerical transformation steps."""

from copy import deepcopy

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.utils.output_specifications import resolve_output_metadata
from exaflow.worker_communication import BadUserInput


def source_output_parameters():
    return {
        "source_variable": specs.ParameterSpecification(
            label="Source variable",
            desc="Variable from which to compute the derived column.",
            types=[specs.ParameterType.TEXT],
            required=True,
            multiple=False,
            enums=specs.ParameterEnumSpecification(
                type=specs.ParameterEnumType.INPUT_VAR_NAMES, source=["variables"]
            ),
        ),
        "new_column_names": specs.ParameterSpecification(
            label="New column names",
            desc="Name assigned to the derived column.",
            types=[specs.ParameterType.TEXT],
            required=True,
            multiple=True,
        ),
    }


def recipe_parameter():
    return specs.ParameterSpecification(
        label="Fitted recipe",
        desc="Stored fitted state to replay instead of fitting again.",
        types=[specs.ParameterType.DICT],
        required=False,
        multiple=False,
    )


def numerical_output(sql_type=specs.InputDataType.REAL):
    return specs.PreprocessingOutputsSpecification(
        new_column_names_param="new_column_names",
        columns=[
            specs.PreprocessingOutputColumnSpecification(
                type=sql_type, stattype=specs.InputDataStatType.NUMERICAL
            )
        ],
    )


def choice_parameter(label, desc, choices):
    return specs.ParameterSpecification(
        label=label,
        desc=desc,
        types=[specs.ParameterType.TEXT],
        required=True,
        multiple=False,
        enums=specs.ParameterEnumSpecification(
            type=specs.ParameterEnumType.LIST, source=list(choices)
        ),
    )


class DerivedNumericalOutput:
    """Mixin for concrete direct PreprocessingStep subclasses.

    Registry discovery inspects direct PreprocessingStep subclasses, so this
    mixin deliberately does not introduce another runtime-step base class.
    The standard base method still owns aggregation injection/output validation.
    """

    _allow_categorical_source = False

    def _value(self, name):
        value = self._params.get(name)
        if value is None:
            return self.get_specification().parameters[name].default
        return value

    @property
    def _source(self):
        source = self._params.get("source_variable")
        if not isinstance(source, str) or not source.strip():
            raise BadUserInput("Provide one source_variable.")
        return source

    @property
    def _output(self):
        names = self._params.get("new_column_names")
        if (
            not isinstance(names, list)
            or len(names) != 1
            or not isinstance(names[0], str)
            or not names[0].strip()
        ):
            raise BadUserInput(
                "new_column_names must contain exactly one nonblank name."
            )
        return names[0]

    def _replay(self, primitive):
        recipe = self._params.get("recipe")
        if recipe is None:
            return None
        if not hasattr(type(primitive), "from_recipe"):
            raise BadUserInput("This method has no fitted recipe to replay.")
        replay = type(primitive).from_recipe(recipe)
        if replay.feature_names_in_ != [self._source] or replay.categorical_features_:
            raise BadUserInput(
                "Recipe must describe exactly the selected numerical source."
            )
        if (
            hasattr(primitive, "output_distribution")
            and replay.output_distribution != primitive.output_distribution
        ):
            raise BadUserInput(
                "Recipe output distribution does not match the selected method."
            )
        if hasattr(primitive, "strategy") and replay.strategy != primitive.strategy:
            raise BadUserInput("Recipe strategy does not match the selected method.")
        if hasattr(primitive, "mode") and replay.mode != primitive.mode:
            raise BadUserInput("Recipe mode does not match the selected method.")
        return replay

    def validate_params(self, *, inputdata, metadata):
        source = self._source
        if source not in inputdata.variables or source not in metadata:
            raise BadUserInput(
                "Source variable must be available in the current input data."
            )
        if not self._allow_categorical_source and (
            metadata[source].get("is_categorical")
            or metadata[source].get("sql_type") not in {"int", "real"}
        ):
            raise BadUserInput("Select a numerical source variable.")
        resolve_output_metadata(self.get_specification(), self._params, metadata)
        try:
            self._replay(self._new_transformer(platform_minimum=1))
        except BadInputError as exc:
            raise BadUserInput(str(exc)) from exc

    def transform_variables(self, *, variables):
        if self._output in variables:
            raise BadUserInput("The derived column name already exists.")
        return [*variables, self._output]

    def transform_metadata(self, *, metadata):
        outputs = resolve_output_metadata(
            self.get_specification(), self._params, metadata
        )
        return {**deepcopy(metadata), **outputs}

    def transform_data(self, *, data, agg_client=None):
        if self._source not in data.columns or self._output in data.columns:
            raise BadInputError("Source must exist and the derived column must be new.")
        # Only quantile fitting consumes the trusted platform minimum. It is
        # obtained from the existing settings layer, never from caller parameters.
        platform_minimum = 1
        if self._params.get("method") in {"quantile_uniform", "quantile_normal"}:
            from exaflow.worker import config as worker_config

            platform_minimum = worker_config.privacy.minimum_row_count
        primitive = self._new_transformer(platform_minimum=platform_minimum)
        replay = self._replay(primitive)
        columns = {"categorical_vars": [], "numerical_vars": [self._source]}
        if replay is None:
            if self.aggregation_server_required() and agg_client is None:
                raise BadInputError(
                    "An aggregation client is required to fit this step."
                )
            primitive.fit(agg_client=agg_client, data=data, **columns)
        else:
            primitive = replay
        result = primitive.transform(data, **columns)
        if self.get_specification().outputs.columns[0].type == specs.InputDataType.INT:
            result = result.astype(int)
        data[self._output] = result[:, 0]
        # This state belongs to this invocation; the standard return path remains
        # dataframe + metadata and does not promise a public recipe response.
        self.fitted_recipe_ = getattr(primitive, "recipe_", None)
        return data
