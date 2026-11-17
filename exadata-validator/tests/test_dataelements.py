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


ENUMS = [{"code": "1", "label": "A"}, {"code": "2", "label": "B"}]
BINARY_ENUMS = [{"code": "0", "label": "No"}, {"code": "1", "label": "Yes"}]
TEXT_ENUMS = [{"code": "a", "label": "A"}, {"code": "b", "label": "B"}]


@pytest.mark.parametrize(
    "variable, error",
    [
        ({"type": "real"}, "invalid 'type'"),
        ({"type": "continuous", "enumerations": ENUMS}, "cannot have enumerations"),
        ({"type": "nominal", "enumerations": ENUMS, "minValue": 0}, "minValue"),
        ({"type": "binary", "enumerations": ENUMS[:1]}, "exactly two enumerations"),
        ({"type": "discrete", "minValue": 0.5, "maxValue": 3}, "whole numbers"),
        ({"type": "nominal", "enumerations": TEXT_ENUMS}, "must be integers"),
        ({"type": "ordinal", "enumerations": TEXT_ENUMS}, "must be integers"),
        ({"type": "binary", "enumerations": TEXT_ENUMS}, "must be integers"),
        ({"type": "binary", "enumerations": ENUMS}, "exactly 0 and 1"),
        ({"type": "nominal", "enumerations": BINARY_ENUMS}, "starting from 1"),
        ({"type": "ordinal", "enumerations": BINARY_ENUMS}, "starting from 1"),
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
        {"type": "ordinal", "enumerations": ENUMS},
        {"type": "binary", "enumerations": BINARY_ENUMS},
        {"type": "nominal", "enumerations": [{"code": "10", "label": "Ten"}]},
    ],
)
def test_valid_cde_type_metadata(variable):
    flatten_cdes({"variables": [{"code": "var", "label": "Var", **variable}]})


@pytest.mark.parametrize("bound", ["min", "max"])
@pytest.mark.parametrize(
    "value", ["0 OR TRUE", "1", True, float("nan"), float("inf"), 10**1000, -(10**1000)]
)
def test_bounds_reject_sql_and_nonnumeric_values(bound, value):
    from validator.dataelements import validate_metadata

    metadata = dict(code="age", label="Age", type="continuous")
    metadata[bound] = value
    with pytest.raises(InvalidDataModelError, match="finite number"):
        validate_metadata("age", metadata)


@pytest.mark.parametrize(
    "bounds",
    [{}, {"min": None, "max": None}, {"min": 0}, {"max": 2.5}, {"min": 0, "max": 2.5}],
)
def test_valid_and_unset_bounds(bounds):
    from validator.dataelements import validate_metadata

    metadata = dict(code="age", label="Age", type="continuous", **bounds)
    validate_metadata("age", metadata)


def test_discrete_bounds_validate_numeric_type_before_whole_number_check():
    from validator.dataelements import validate_metadata

    base = dict(code="age", label="Age", type="discrete")
    validate_metadata("age", {**base, "min": None})
    with pytest.raises(InvalidDataModelError, match="finite number"):
        validate_metadata("age", {**base, "min": "abc"})
    with pytest.raises(InvalidDataModelError, match="whole numbers"):
        validate_metadata("age", {**base, "min": 1.5})


def test_noncanonical_and_duplicate_enumeration_codes_are_rejected():
    base = {"code": "var", "label": "Var", "type": "nominal"}
    with pytest.raises(InvalidDataModelError, match="must be integers"):
        flatten_cdes(
            {"variables": [{**base, "enumerations": [{"code": "01", "label": "A"}]}]}
        )
    with pytest.raises(InvalidDataModelError, match="duplicate enumeration codes"):
        flatten_cdes(
            {
                "variables": [
                    {
                        **base,
                        "enumerations": [
                            {"code": "1", "label": "A"},
                            {"code": "1", "label": "B"},
                        ],
                    }
                ]
            }
        )


def test_dataset_codes_must_be_integers():
    variable = {"code": "dataset", "label": "Dataset", "type": "nominal"}
    variable["enumerations"] = TEXT_ENUMS
    with pytest.raises(InvalidDataModelError, match="must be integers"):
        flatten_cdes({"variables": [variable]})


@pytest.mark.parametrize("code", ["visitid"])
def test_identifier_columns_must_have_integer_codes(code):
    variable = {
        "code": code,
        "label": code,
        "type": "nominal",
        "enumerations": TEXT_ENUMS,
    }
    with pytest.raises(InvalidDataModelError, match="must be integers"):
        flatten_cdes({"variables": [variable]})


@pytest.mark.parametrize(
    "variable",
    [
        {"code": "subjectid", "label": "subjectid", "type": "discrete"},
        {
            "code": "visitid",
            "label": "visitid",
            "type": "nominal",
            "enumerations": [
                {"code": "1", "label": "BL"},
                {"code": "2", "label": "FL1"},
            ],
        },
    ],
    ids=["discrete-subjectid", "nominal-visitid"],
)
def test_identifier_columns_are_accepted(variable):
    flatten_cdes({"variables": [variable]})


@pytest.mark.parametrize("cde_type", ["nominal", "ordinal", "binary"])
@pytest.mark.parametrize("code", ["var", "visitid"])
@pytest.mark.parametrize("enumerations", [None, []])
def test_categorical_cde_without_enumerations_is_rejected(cde_type, code, enumerations):
    variable = {"code": code, "label": code, "type": cde_type}
    if enumerations is not None:
        variable["enumerations"] = enumerations
    with pytest.raises(InvalidDataModelError, match="must have enumerations"):
        flatten_cdes({"variables": [variable]})
