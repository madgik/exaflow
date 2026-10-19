"""Resolve declared preprocessing columns without inspecting observation data."""

from numbers import Integral

from exaflow.algorithms.specifications import InputDataStatType
from exaflow.algorithms.specifications import PreprocessingOutputEnumerationSource


def resolve_output_metadata(specification, parameters, existing_metadata):
    # worker_communication imports exaflow, which imports the preprocessing base.
    from exaflow.worker_communication import BadUserInput

    outputs = specification.outputs
    if outputs is None:
        return {}
    names = parameters.get(outputs.new_column_names_param)
    if (
        not isinstance(names, list)
        or len(names) != len(outputs.columns)
        or any(not isinstance(name, str) or not name.strip() for name in names)
    ):
        raise BadUserInput(
            f"'{outputs.new_column_names_param}' must contain exactly "
            f"{len(outputs.columns)} nonblank column names."
        )
    if len(set(names)) != len(names):
        raise BadUserInput("New column names must be unique.")
    for name in names:
        if name in existing_metadata:
            raise BadUserInput(
                f"Preprocessing step '{specification.name}' cannot create CDE "
                f"'{name}' because it already exists."
            )

    def value(parameter_name):
        return parameters.get(
            parameter_name, specification.parameters[parameter_name].default
        )

    resolved = {}
    for name, column in zip(names, outputs.columns):
        nominal = column.stattype == InputDataStatType.NOMINAL
        metadata = {
            "code": name,
            "label": name,
            "sql_type": column.type.value,
            "is_categorical": nominal,
        }
        if nominal:
            enums = column.enumerations
            source = value(enums.source_parameter_name)
            if enums.source_type == PreprocessingOutputEnumerationSource.DICT_KEYS:
                if not isinstance(source, dict) or not source:
                    raise BadUserInput(
                        f"'{enums.source_parameter_name}' must be a nonempty dictionary."
                    )
                categories = list(source)
                if enums.additional_value_parameter_name is not None:
                    additional = value(enums.additional_value_parameter_name)
                    if additional is not None:
                        categories.append(additional)
            else:
                prefix = value(enums.prefix_parameter_name)
                # Protobuf Struct transports integer parameters as doubles.
                if isinstance(source, float) and source.is_integer():
                    source = int(source)
                if (
                    isinstance(source, bool)
                    or not isinstance(source, Integral)
                    or source < 1
                ):
                    raise BadUserInput(
                        f"'{enums.source_parameter_name}' must be a positive integer."
                    )
                if not isinstance(prefix, str):
                    raise BadUserInput("Enumeration prefix must be text.")
                categories = [f"{prefix}{index}" for index in range(1, source + 1)]
            if any(
                not isinstance(category, str) or not category.strip()
                for category in categories
            ):
                raise BadUserInput("Enumeration values must be nonblank strings.")
            metadata["enumerations"] = {category: category for category in categories}
        resolved[name] = metadata
    return resolved


def validate_output_metadata(expected, metadata, variables):
    names = list(expected)
    if [name for name in variables if name in expected] != names:
        raise ValueError("Transformed columns do not match the declared output order.")
    for name, declared in expected.items():
        actual = metadata.get(name, {})
        if any(actual.get(key) != value for key, value in declared.items()):
            raise ValueError(
                f"Metadata for output column '{name}' contradicts its specification."
            )
        if not declared["is_categorical"] and actual.get("enumerations"):
            raise ValueError(
                f"Numerical output column '{name}' cannot have enumerations."
            )
