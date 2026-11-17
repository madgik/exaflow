import json
from pathlib import Path

import pytest
from validator.duckdb_validator import DuckDBDatasetValidator
from validator.exceptions import InvalidDatasetError

TEST_DIR = Path(__file__).resolve().parent
DATA_DIR = TEST_DIR / "data"


def test_validate_csv_success(success_metadata):
    csv_path = DATA_DIR / "success" / "data_model_v_1_0" / "dataset.csv"
    DuckDBDatasetValidator(success_metadata).validate_csv(csv_path)


def test_validate_csv_invalid_type(fail_metadata):
    csv_path = DATA_DIR / "fail" / "data_model_v_1_0" / "invalid_type1.csv"
    with pytest.raises(
        InvalidDatasetError, match="Column 'var3' has invalid continuous values"
    ):
        DuckDBDatasetValidator(fail_metadata).validate_csv(csv_path)


def test_validate_csv_missing_dataset_column(fail_metadata):
    csv_path = DATA_DIR / "fail" / "data_model_v_1_0" / "missing_column_dataset.csv"
    with pytest.raises(InvalidDatasetError, match="'dataset' column"):
        DuckDBDatasetValidator(fail_metadata).validate_csv(csv_path)


def test_validate_csv_invalid_enum(fail_metadata):
    csv_path = DATA_DIR / "fail" / "data_model_v_1_0" / "invalid_enum.csv"
    with pytest.raises(InvalidDatasetError, match="has invalid categorical value"):
        DuckDBDatasetValidator(fail_metadata).validate_csv(csv_path)


def test_discrete_column_rejects_decimal_values(tmp_path):
    metadata = {
        "variables": [
            {
                "code": "dataset",
                "label": "Dataset",
                "type": "nominal",
                "enumerations": [{"code": "1", "label": "D"}],
            },
            {"code": "score", "label": "Score", "type": "discrete"},
        ]
    }
    path = tmp_path / "discrete.csv"
    path.write_text("dataset,score\n1,1.5\n", encoding="utf-8")
    issues = DuckDBDatasetValidator(metadata).validate_csv(path, report_all=True)
    assert "types.invalid" in {issue.rule for issue in issues}


def test_validate_csv_unknown_column_suggests_closest_cde(success_metadata, tmp_path):
    csv_path = tmp_path / "dataset.csv"
    csv_path.write_text(
        "dataset,vra1\n1,10\n",
        encoding="utf-8",
    )

    with pytest.raises(InvalidDatasetError, match="Did you mean 'var1'"):
        DuckDBDatasetValidator(success_metadata).validate_csv(csv_path)


def test_validate_csv_longitudinal_duplicate_pair():
    metadata = json.loads(
        (
            DATA_DIR / "fail" / "data_model_longitudinal_v_1_0" / "CDEsMetadata.json"
        ).read_text(encoding="utf-8")
    )
    csv_path = DATA_DIR / "fail" / "data_model_longitudinal_v_1_0" / "dataset.csv"

    with pytest.raises(InvalidDatasetError, match=r"duplicate \(visitid, subjectid\)"):
        DuckDBDatasetValidator(metadata).validate_csv(csv_path)


@pytest.mark.parametrize("report_all", [False, True])
def test_bound_parameters_preserve_range_checks(tmp_path, report_all):
    metadata = {
        "variables": [
            {
                "code": "dataset",
                "label": "Dataset",
                "type": "nominal",
                "enumerations": [{"code": "1", "label": "D"}],
            },
            {
                "code": "age",
                "label": "Age",
                "type": "continuous",
                "minValue": 1,
                "maxValue": 10.5,
            },
        ]
    }
    path = tmp_path / "bounds.csv"
    path.write_text("dataset,age\n1,0\n1,5\n1,12\n")
    validator = DuckDBDatasetValidator(metadata)
    if report_all:
        issues = validator.validate_csv(path, report_all=True)
        assert {issue.rule for issue in issues} == {"range.min", "range.max"}
    else:
        with pytest.raises(InvalidDatasetError):
            validator.validate_csv(path)


def _integer_code_metadata():
    def enums(*codes):
        return [{"code": code, "label": f"L{code}"} for code in codes]

    return {
        "variables": [
            {
                "code": "dataset",
                "label": "Dataset",
                "type": "nominal",
                "enumerations": enums("1"),
            },
            {
                "code": "flag",
                "label": "Flag",
                "type": "binary",
                "enumerations": enums("0", "1"),
            },
            {
                "code": "grade",
                "label": "Grade",
                "type": "ordinal",
                "enumerations": enums("1", "2", "3"),
            },
            {
                "code": "kind",
                "label": "Kind",
                "type": "nominal",
                "enumerations": enums("1", "2"),
            },
        ]
    }


def test_integer_codes_are_accepted(tmp_path):
    path = tmp_path / "ok.csv"
    path.write_text("dataset,flag,grade,kind\n1,0,1,1\n1,1,3,2\n1,,,\n")
    DuckDBDatasetValidator(_integer_code_metadata()).validate_csv(path)


@pytest.mark.parametrize(
    "row, rule",
    [
        ("1,yes,1,1", "types.invalid"),
        ("1,0,low,1", "types.invalid"),
        ("1,0,1,a", "types.invalid"),
        ("a,0,1,1", "types.invalid"),
        ("1,0,1.5,1", "types.invalid"),
        ("1,2,1,1", "enum.invalid"),
        ("1,0,4,1", "enum.invalid"),
        ("1,0,1,3", "enum.invalid"),
        ("3,0,1,1", "enum.invalid"),
    ],
)
def test_non_integer_or_undeclared_codes_are_rejected(tmp_path, row, rule):
    path = tmp_path / "bad.csv"
    path.write_text(f"dataset,flag,grade,kind\n{row}\n")
    issues = DuckDBDatasetValidator(_integer_code_metadata()).validate_csv(
        path, report_all=True
    )
    assert rule in {issue.rule for issue in issues}


def _identifier_metadata():
    def enums(*codes):
        return [{"code": code, "label": f"L{code}"} for code in codes]

    return {
        "longitudinal": True,
        "variables": [
            {
                "code": "dataset",
                "label": "Dataset",
                "type": "nominal",
                "enumerations": enums("1"),
            },
            {
                "code": "subjectid",
                "label": "Subject",
                "type": "discrete",
            },
            {
                "code": "visitid",
                "label": "Visit",
                "type": "nominal",
                "enumerations": enums("1", "2"),
            },
        ],
    }


def test_integer_subject_and_visit_codes_are_accepted(tmp_path):
    path = tmp_path / "ok.csv"
    path.write_text("dataset,subjectid,visitid\n1,1,1\n1,1,2\n1,2,1\n")
    DuckDBDatasetValidator(_identifier_metadata()).validate_csv(path)


@pytest.mark.parametrize(
    "row",
    ["1,subject1,1", "1,1,BL", "1,1.5,1", "1,1,FL1", "1,1,3"],
)
def test_text_or_undeclared_subject_and_visit_codes_are_rejected(tmp_path, row):
    path = tmp_path / "bad.csv"
    path.write_text(f"dataset,subjectid,visitid\n{row}\n")
    issues = DuckDBDatasetValidator(_identifier_metadata()).validate_csv(
        path, report_all=True
    )
    assert {issue.rule for issue in issues} & {"types.invalid", "enum.invalid"}
