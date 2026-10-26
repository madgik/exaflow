import pytest
from validator.dataelements import CommonDataElement
from validator.dataelements import flatten_cdes
from validator.dataelements import validate_dataset_present_on_cdes_with_proper_format
from validator.dataelements import validate_longitudinal_data_model
from validator.exceptions import InvalidDataModelError


def test_flatten_cdes_success(success_metadata):
    cdes = flatten_cdes(success_metadata)
    assert len(cdes) == 6
    assert all(isinstance(cde, CommonDataElement) for cde in cdes)


def test_dataset_cde_validation_success(success_metadata):
    cdes = flatten_cdes(success_metadata)
    validate_dataset_present_on_cdes_with_proper_format(cdes)


def test_dataset_cde_missing_raises(success_metadata):
    cdes = [cde for cde in flatten_cdes(success_metadata) if cde.code != "dataset"]
    with pytest.raises(InvalidDataModelError, match="There is no 'dataset' CDE"):
        validate_dataset_present_on_cdes_with_proper_format(cdes)


def test_longitudinal_validation_requires_subjectid_and_visitid():
    with pytest.raises(InvalidDataModelError, match="subjectid"):
        validate_longitudinal_data_model([])


ENUMS = [{"code": "a", "label": "A"}, {"code": "b", "label": "B"}]


@pytest.mark.parametrize(
    "variable, error",
    [
        ({"type": "real"}, "invalid 'type'"),
        ({"type": "continuous", "enumerations": ENUMS}, "cannot have enumerations"),
        ({"type": "nominal", "enumerations": ENUMS, "minValue": 0}, "minValue"),
        ({"type": "binary", "enumerations": ENUMS[:1]}, "exactly two enumerations"),
        ({"type": "discrete", "minValue": 0.5, "maxValue": 3}, "whole numbers"),
    ],
)
def test_invalid_cde_type_metadata_raises(variable, error):
    variable = {"code": "var", "label": "Var", **variable}
    with pytest.raises(InvalidDataModelError, match=error):
        flatten_cdes({"variables": [variable]})


@pytest.mark.parametrize(
    "variable",
    [
        {"type": "continuous", "minValue": 0.5, "maxValue": 3.5},
        {"type": "discrete", "minValue": 0, "maxValue": 3},
        {"type": "nominal"},
        {"type": "ordinal", "enumerations": ENUMS},
        {"type": "binary", "enumerations": ENUMS},
    ],
)
def test_valid_cde_type_metadata(variable):
    flatten_cdes({"variables": [{"code": "var", "label": "Var", **variable}]})
