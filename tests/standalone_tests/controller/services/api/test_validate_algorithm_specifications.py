import pytest
from pydantic import ValidationError

from exaflow.algorithms.specifications import AlgorithmSpecification
from exaflow.algorithms.specifications import InputDataSpecification
from exaflow.algorithms.specifications import InputDataStatType
from exaflow.algorithms.specifications import InputDataType
from exaflow.algorithms.specifications import ParameterDictValueType
from exaflow.algorithms.specifications import ParameterEnumSpecification
from exaflow.algorithms.specifications import ParameterEnumType
from exaflow.algorithms.specifications import ParameterSpecification
from exaflow.algorithms.specifications import ParameterType
from exaflow.algorithms.specifications import PreprocessingStepSpecification
from exaflow.controller.services.api.algorithm_spec_dtos import (
    _convert_algorithm_specification_to_dto,
)
from exaflow.controller.services.api.algorithm_spec_dtos import (
    _convert_transformer_specification_to_dto,
)
from exaflow.controller.services.api.algorithm_spec_dtos import (
    inputdata_specification_dto,
)
from exaflow.controller.services.specifications import Specifications


def _input_spec(label="Outcome"):
    return InputDataSpecification(
        label=label,
        desc=label,
        types=[InputDataType.TEXT],
        stattypes=[InputDataStatType.NOMINAL],
        required=True,
        min_count=1,
        max_count=1,
    )


def _sample_algorithm_spec(**updates):
    values = {
        "name": "sample_algo",
        "desc": "sample",
        "documentation": "sample",
        "label": "sample_algo",
        "enabled": True,
        "y": _input_spec("Outcome"),
        "x": _input_spec("Features"),
    }
    values.update(updates)
    return AlgorithmSpecification(**values)


def _sample_preprocessing_spec(name="sample_preprocessing", *, enabled=True, **updates):
    values = {
        "name": name,
        "desc": name,
        "documentation": name,
        "label": name,
        "enabled": enabled,
    }
    values.update(updates)
    return PreprocessingStepSpecification(**values)


def test_algorithm_spec_required_preprocessing_defaults_to_empty_list():
    spec = _sample_algorithm_spec()

    assert spec.required_preprocessing == []


def test_algorithm_spec_dto_exposes_flat_xy_without_inputdata_or_preprocessing():
    spec = _sample_algorithm_spec(
        required_preprocessing=["sample_preprocessing"],
        requires_validation_datasets=True,
    )

    dto = _convert_algorithm_specification_to_dto(spec)

    assert dto.y.label == "Outcome"
    assert dto.x.label == "Features"
    assert dto.requires_validation_datasets is True
    assert dto.required_preprocessing == ["sample_preprocessing"]
    assert not hasattr(dto, "inputdata")
    assert not hasattr(dto, "preprocessing")


def test_inputdata_specification_dto_exposes_source_variables_and_filters():
    spec = inputdata_specification_dto.root

    assert spec.variables.required is True
    assert spec.filters.required is False
    assert hasattr(spec, "filters")
    assert not hasattr(spec, "filter")


def test_preprocessing_spec_dto_exposes_output_columns():
    from exaflow.algorithms.exareme3.feature_engineering.categorical_column_creator import (
        CategoricalColumnCreator,
    )

    dto = _convert_transformer_specification_to_dto(
        CategoricalColumnCreator.get_specification()
    )
    serialized = dto.model_dump(mode="json")
    assert serialized["outputs"] == {
        "new_column_names_param": "new_column_names",
        "columns": [
            {
                "type": "text",
                "stattype": "nominal",
                "column_count_param": None,
                "enumerations": {
                    "source_parameter_name": "rules",
                    "source_type": "dict_keys",
                    "additional_value_parameter_name": "default_enumeration",
                    "prefix_parameter_name": None,
                },
            }
        ],
    }
    assert "output" not in serialized


def test_parameter_dict_value_type_supports_filter():
    parameter = ParameterSpecification(
        label="Enumeration filters",
        desc="Dictionary where each value is a filter.",
        types=[ParameterType.DICT],
        required=True,
        multiple=False,
        dict_values_type=ParameterDictValueType.FILTER,
    )

    assert parameter.dict_values_type == ParameterDictValueType.FILTER


def test_preprocessing_input_var_names_source_must_be_variables():
    with pytest.raises(ValidationError, match="Allowed sources are .*variables"):
        _sample_preprocessing_spec(
            parameters={
                "strategies": ParameterSpecification(
                    label="Strategies",
                    desc="Strategies.",
                    types=[ParameterType.DICT],
                    required=True,
                    multiple=False,
                    dict_keys_enums=ParameterEnumSpecification(
                        type=ParameterEnumType.INPUT_VAR_NAMES,
                        source=["x"],
                    ),
                )
            }
        )


def test_preprocessing_input_var_names_source_accepts_variables():
    spec = _sample_preprocessing_spec(
        parameters={
            "strategies": ParameterSpecification(
                label="Strategies",
                desc="Strategies.",
                types=[ParameterType.DICT],
                required=True,
                multiple=False,
                dict_keys_enums=ParameterEnumSpecification(
                    type=ParameterEnumType.INPUT_VAR_NAMES,
                    source=["variables"],
                ),
            )
        }
    )

    assert spec.parameters["strategies"].dict_keys_enums.source == ["variables"]


def test_specifications_raise_when_required_preprocessing_is_missing_or_disabled():
    specifications = Specifications.__new__(Specifications)
    specifications.enabled_algorithms = {
        "sample_algo": _sample_algorithm_spec(
            required_preprocessing=["sample_preprocessing"]
        )
    }
    specifications.enabled_preprocessing_steps = {}

    with pytest.raises(
        ValueError,
        match=(
            "Algorithm 'sample_algo' requires preprocessing steps that are not "
            "enabled: .*sample_preprocessing.*"
        ),
    ):
        specifications._validate_required_preprocessing()


def test_algorithm_parameter_input_var_cde_enums_source_must_be_x_or_y():
    with pytest.raises(ValidationError, match="supports only 'x' or 'y' as source"):
        _sample_algorithm_spec(
            parameters={
                "sample_param": ParameterSpecification(
                    label="sample_label",
                    desc="sample",
                    types=[ParameterType.TEXT],
                    required=True,
                    multiple=False,
                    enums=ParameterEnumSpecification(
                        type=ParameterEnumType.INPUT_VAR_CDE_ENUMS,
                        source=["variables"],
                    ),
                )
            }
        )
