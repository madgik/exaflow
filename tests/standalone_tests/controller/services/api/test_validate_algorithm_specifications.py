import pytest
from pydantic import ValidationError

from exaflow.algorithms.specifications import AlgorithmSpecification
from exaflow.algorithms.specifications import InputDataSpecification
from exaflow.algorithms.specifications import InputDataSpecifications
from exaflow.algorithms.specifications import InputDataStatType
from exaflow.algorithms.specifications import InputDataType
from exaflow.algorithms.specifications import ParameterDictValueType
from exaflow.algorithms.specifications import ParameterEnumSpecification
from exaflow.algorithms.specifications import ParameterEnumType
from exaflow.algorithms.specifications import ParameterSpecification
from exaflow.algorithms.specifications import ParameterType


def test_validate_parameter_spec_input_var_CDE_enums_source_is_x_or_y():
    exception_type = ValidationError
    exception_message = (
        ".*In algorithm 'sample_algo', parameter 'sample_label' has enums type 'input_var_CDE_enums' "
        "that supports only 'x' or 'y' as source. Value given: 'not_x_or_y'.*"
    )
    with pytest.raises(exception_type, match=exception_message):
        AlgorithmSpecification(
            name="sample_algo",
            desc="sample",
            documentation="sample",
            label="sample_algo",
            enabled=True,
            inputdata=InputDataSpecifications(
                y=InputDataSpecification(
                    label="y",
                    desc="y",
                    types=[InputDataType.TEXT],
                    stattypes=[InputDataStatType.NOMINAL],
                    required=True,
                    multiple=False,
                )
            ),
            parameters={
                "inputdata_cde_enum_param": ParameterSpecification(
                    label="sample_label",
                    desc="sample",
                    types=[ParameterType.TEXT],
                    required=False,
                    multiple=False,
                    enums=ParameterEnumSpecification(
                        type=ParameterEnumType.INPUT_VAR_CDE_ENUMS,
                        source=["not_x_or_y"],
                    ),
                ),
            },
        )


def test_validate_parameter_spec_input_var_CDE_enums_multiple_false():
    exception_type = ValidationError
    exception_message = (
        ".*In algorithm 'sample_algo', parameter 'sample_label' has enums type 'input_var_CDE_enums' "
        "that doesn't support 'multiple=True', in the parameter.*"
    )
    with pytest.raises(exception_type, match=exception_message):
        AlgorithmSpecification(
            name="sample_algo",
            desc="sample",
            documentation="sample",
            label="sample_algo",
            enabled=True,
            inputdata=InputDataSpecifications(
                y=InputDataSpecification(
                    label="y",
                    desc="y",
                    types=[InputDataType.TEXT],
                    stattypes=[InputDataStatType.NOMINAL],
                    required=True,
                    multiple=False,
                )
            ),
            parameters={
                "inputdata_cde_enum_param": ParameterSpecification(
                    label="sample_label",
                    desc="sample",
                    types=[ParameterType.TEXT],
                    required=False,
                    multiple=True,
                    enums=ParameterEnumSpecification(
                        type=ParameterEnumType.INPUT_VAR_CDE_ENUMS, source=["y"]
                    ),
                ),
            },
        )


def test_validate_parameter_spec_input_var_CDE_enums_inputdata_has_multiple_false():
    exception_type = ValidationError
    exception_message = (
        ".* In algorithm 'sample_algo', parameter 'sample_label' has enums type 'input_var_CDE_enums' "
        "that doesn't support 'multiple=True' in it's linked inputdata var 'y'.*"
    )
    with pytest.raises(exception_type, match=exception_message):
        AlgorithmSpecification(
            name="sample_algo",
            desc="sample",
            documentation="sample",
            label="sample_algo",
            enabled=True,
            inputdata=InputDataSpecifications(
                y=InputDataSpecification(
                    label="y",
                    desc="y",
                    types=[InputDataType.TEXT],
                    stattypes=[InputDataStatType.NOMINAL],
                    required=True,
                    multiple=True,
                )
            ),
            parameters={
                "inputdata_cde_enum_param": ParameterSpecification(
                    label="sample_label",
                    desc="sample",
                    types=[ParameterType.TEXT],
                    required=False,
                    multiple=False,
                    enums=ParameterEnumSpecification(
                        type=ParameterEnumType.INPUT_VAR_CDE_ENUMS, source=["y"]
                    ),
                ),
            },
        )


def test_validate_parameter_spec_input_var_names_type_must_be_text():
    exception_type = ValidationError
    exception_message = (
        """.* In algorithm 'sample_algo', parameter 'sample_label' has enums type 'input_var_names' """
        """that supports ONLY '.*' but the 'types' provided were .*"""
    )
    with pytest.raises(exception_type, match=exception_message):
        AlgorithmSpecification(
            name="sample_algo",
            desc="sample",
            documentation="sample",
            label="sample_algo",
            enabled=True,
            inputdata=InputDataSpecifications(
                y=InputDataSpecification(
                    label="y",
                    desc="y",
                    types=[InputDataType.TEXT],
                    stattypes=[InputDataStatType.NOMINAL],
                    required=True,
                    multiple=True,
                )
            ),
            parameters={
                "input_var_names_enum_param": ParameterSpecification(
                    label="sample_label",
                    desc="sample",
                    types=[ParameterType.INT],
                    required=False,
                    multiple=False,
                    enums=ParameterEnumSpecification(
                        type=ParameterEnumType.INPUT_VAR_NAMES, source=["y"]
                    ),
                ),
            },
        )


def test_validate_parameter_spec_input_var_CDE_enums_only_one_value():
    exception_type = ValidationError
    exception_message = (
        ".*In algorithm 'sample_algo', parameter 'sample_label' has enums type 'input_var_CDE_enums' "
        "that supports only one value."
    )
    with pytest.raises(exception_type, match=exception_message):
        AlgorithmSpecification(
            name="sample_algo",
            desc="sample",
            documentation="sample",
            label="sample_algo",
            enabled=True,
            inputdata=InputDataSpecifications(
                y=InputDataSpecification(
                    label="y",
                    desc="y",
                    types=[InputDataType.TEXT],
                    stattypes=[InputDataStatType.NOMINAL],
                    required=True,
                    multiple=False,
                )
            ),
            parameters={
                "inputdata_cde_enum_param": ParameterSpecification(
                    label="sample_label",
                    desc="sample",
                    types=[ParameterType.TEXT],
                    required=False,
                    multiple=True,
                    enums=ParameterEnumSpecification(
                        type=ParameterEnumType.INPUT_VAR_CDE_ENUMS,
                        source=["y", "second_value"],
                    ),
                ),
            },
        )


def test_validate_parameter_spec_fixed_var_CDE_enums_only_one_value():
    exception_type = ValidationError
    exception_message = (
        ".*In algorithm 'sample_algo', parameter 'sample_label' has enums type 'fixed_var_CDE_enums' "
        "that supports only one value."
    )
    with pytest.raises(exception_type, match=exception_message):
        AlgorithmSpecification(
            name="sample_algo",
            desc="sample",
            documentation="sample",
            label="sample_algo",
            enabled=True,
            inputdata=InputDataSpecifications(
                y=InputDataSpecification(
                    label="y",
                    desc="y",
                    types=[InputDataType.TEXT],
                    stattypes=[InputDataStatType.NOMINAL],
                    required=True,
                    multiple=False,
                )
            ),
            parameters={
                "inputdata_cde_enum_param": ParameterSpecification(
                    label="sample_label",
                    desc="sample",
                    types=[ParameterType.TEXT],
                    required=False,
                    multiple=True,
                    enums=ParameterEnumSpecification(
                        type=ParameterEnumType.FIXED_VAR_CDE_ENUMS,
                        source=["y", "second_value"],
                    ),
                ),
            },
        )


def test_validate_parameter_dict_type_given_with_other_type():
    exception_type = ValidationError
    exception_message = (
        ".*In algorithm 'sample_algo', parameter 'sample_label' cannot use 'dict' type combined"
        " with other types. Types provided: .* "
    )
    with pytest.raises(exception_type, match=exception_message):
        AlgorithmSpecification(
            name="sample_algo",
            desc="sample",
            documentation="sample",
            label="sample_algo",
            enabled=True,
            inputdata=InputDataSpecifications(
                y=InputDataSpecification(
                    label="y",
                    desc="y",
                    types=[InputDataType.TEXT],
                    stattypes=[InputDataStatType.NOMINAL],
                    required=True,
                    multiple=False,
                )
            ),
            parameters={
                "dict_and_text_types_param": ParameterSpecification(
                    label="sample_label",
                    desc="sample",
                    types=[ParameterType.DICT, ParameterType.TEXT],
                    required=False,
                    multiple=False,
                ),
            },
        )


def test_validate_parameter_property_dict_keys_enums_can_only_be_given_with_type_dict():
    exception_type = ValidationError
    exception_message = (
        ".*In algorithm 'sample_algo', parameter 'sample_label' has the property 'dict_keys_enums' "
        "but the allowed 'types' is not 'dict'."
    )
    with pytest.raises(exception_type, match=exception_message):
        AlgorithmSpecification(
            name="sample_algo",
            desc="sample",
            documentation="sample",
            label="sample_algo",
            enabled=True,
            inputdata=InputDataSpecifications(
                y=InputDataSpecification(
                    label="y",
                    desc="y",
                    types=[InputDataType.TEXT],
                    stattypes=[InputDataStatType.NOMINAL],
                    required=True,
                    multiple=False,
                )
            ),
            parameters={
                "dict_keys_enums_param": ParameterSpecification(
                    label="sample_label",
                    desc="sample",
                    types=[ParameterType.TEXT],
                    required=False,
                    multiple=False,
                    dict_keys_enums=ParameterEnumSpecification(
                        type=ParameterEnumType.LIST, source=["sample_enum"]
                    ),
                ),
            },
        )


def test_validate_parameter_property_dict_values_enums_can_only_be_given_with_type_dict():
    exception_type = ValidationError
    exception_message = (
        ".*In algorithm 'sample_algo', parameter 'sample_label' has the property 'dict_values_enums' "
        "but the allowed 'types' is not 'dict'."
    )
    with pytest.raises(exception_type, match=exception_message):
        AlgorithmSpecification(
            name="sample_algo",
            desc="sample",
            documentation="sample",
            label="sample_algo",
            enabled=True,
            inputdata=InputDataSpecifications(
                y=InputDataSpecification(
                    label="y",
                    desc="y",
                    types=[InputDataType.TEXT],
                    stattypes=[InputDataStatType.NOMINAL],
                    required=True,
                    multiple=False,
                )
            ),
            parameters={
                "dict_values_enums_param": ParameterSpecification(
                    label="sample_label",
                    desc="sample",
                    types=[ParameterType.TEXT],
                    required=False,
                    multiple=False,
                    dict_values_enums=ParameterEnumSpecification(
                        type=ParameterEnumType.LIST, source=["sample_enum"]
                    ),
                ),
            },
        )


def test_validate_parameter_property_dict_values_type_can_only_be_given_with_type_dict():
    exception_type = ValidationError
    exception_message = (
        ".*In algorithm 'sample_algo', parameter 'sample_label' has the property 'dict_values_type' "
        "but the allowed 'types' is not 'dict'."
    )
    with pytest.raises(exception_type, match=exception_message):
        AlgorithmSpecification(
            name="sample_algo",
            desc="sample",
            documentation="sample",
            label="sample_algo",
            enabled=True,
            inputdata=InputDataSpecifications(
                y=InputDataSpecification(
                    label="y",
                    desc="y",
                    types=[InputDataType.TEXT],
                    stattypes=[InputDataStatType.NOMINAL],
                    required=True,
                    multiple=False,
                )
            ),
            parameters={
                "dict_values_type_param": ParameterSpecification(
                    label="sample_label",
                    desc="sample",
                    types=[ParameterType.TEXT],
                    required=False,
                    multiple=False,
                    dict_values_type=ParameterDictValueType.REAL,
                ),
            },
        )


def test_validate_parameter_property_enums_given_with_type_dict():
    exception_type = ValidationError
    exception_message = (
        ".*In algorithm 'sample_algo', parameter 'sample_label' has the property 'enums' "
        "but since the 'types' is 'dict', you should use 'dict_keys_enums' and 'dict_values_enums'."
    )
    with pytest.raises(exception_type, match=exception_message):
        AlgorithmSpecification(
            name="sample_algo",
            desc="sample",
            documentation="sample",
            label="sample_algo",
            enabled=True,
            inputdata=InputDataSpecifications(
                y=InputDataSpecification(
                    label="y",
                    desc="y",
                    types=[InputDataType.TEXT],
                    stattypes=[InputDataStatType.NOMINAL],
                    required=True,
                    multiple=False,
                )
            ),
            parameters={
                "dict_type_param": ParameterSpecification(
                    label="sample_label",
                    desc="sample",
                    types=[ParameterType.DICT],
                    required=False,
                    multiple=False,
                    enums=ParameterEnumSpecification(
                        type=ParameterEnumType.LIST, source=["sample_enum"]
                    ),
                ),
            },
        )


def test_validate_inputdata_min_cannot_be_negative():
    exception_type = ValidationError
    exception_message = ".*'min' should be greater than or equal to 0.*"
    with pytest.raises(exception_type, match=exception_message):
        InputDataSpecification(
            label="x",
            desc="x",
            types=[InputDataType.REAL],
            stattypes=[InputDataStatType.NUMERICAL],
            required=True,
            multiple=True,
            min=-1,
        )


def test_validate_inputdata_max_cannot_be_negative():
    exception_type = ValidationError
    exception_message = ".*'max' should be greater than or equal to 0.*"
    with pytest.raises(exception_type, match=exception_message):
        InputDataSpecification(
            label="x",
            desc="x",
            types=[InputDataType.REAL],
            stattypes=[InputDataStatType.NUMERICAL],
            required=True,
            multiple=True,
            max=-1,
        )


def test_validate_inputdata_min_cannot_exceed_max():
    exception_type = ValidationError
    exception_message = ".*'min' cannot be greater than 'max'.*"
    with pytest.raises(exception_type, match=exception_message):
        InputDataSpecification(
            label="x",
            desc="x",
            types=[InputDataType.REAL],
            stattypes=[InputDataStatType.NUMERICAL],
            required=True,
            multiple=True,
            min=3,
            max=2,
        )


def test_validate_inputdata_multiple_false_incompatible_with_min_gt_one():
    exception_type = ValidationError
    exception_message = ".*'multiple=False' is incompatible with 'min' greater than 1.*"
    with pytest.raises(exception_type, match=exception_message):
        InputDataSpecification(
            label="x",
            desc="x",
            types=[InputDataType.REAL],
            stattypes=[InputDataStatType.NUMERICAL],
            required=True,
            multiple=False,
            min=2,
        )


def test_validate_inputdata_multiple_false_incompatible_with_max_gt_one():
    exception_type = ValidationError
    exception_message = ".*'multiple=False' is incompatible with 'max' greater than 1.*"
    with pytest.raises(exception_type, match=exception_message):
        InputDataSpecification(
            label="x",
            desc="x",
            types=[InputDataType.REAL],
            stattypes=[InputDataStatType.NUMERICAL],
            required=True,
            multiple=False,
            max=2,
        )
