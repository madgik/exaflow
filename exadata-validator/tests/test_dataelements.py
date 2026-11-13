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


@pytest.mark.parametrize("bound", ["min", "max"])
@pytest.mark.parametrize(
    "value", ["0 OR TRUE", "1", True, float("nan"), float("inf"), 10**1000, -(10**1000)]
)
def test_bounds_reject_sql_and_nonnumeric_values(bound, value):
    from validator.dataelements import validate_metadata

    metadata = dict(
        code="age", label="Age", type="real", sql_type="real", is_categorical=False
    )
    metadata[bound] = value
    with pytest.raises(InvalidDataModelError, match="finite number"):
        validate_metadata("age", metadata)


@pytest.mark.parametrize(
    "bounds",
    [{}, {"min": None, "max": None}, {"min": 0}, {"max": 2.5}, {"min": 0, "max": 2.5}],
)
def test_valid_and_unset_bounds(bounds):
    from validator.dataelements import validate_metadata

    metadata = dict(
        code="age",
        label="Age",
        type="real",
        sql_type="real",
        is_categorical=False,
        **bounds,
    )
    validate_metadata("age", metadata)
