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
        InvalidDatasetError, match="Column 'var3' has invalid real values"
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


def test_validate_csv_unknown_column_suggests_closest_cde(success_metadata, tmp_path):
    csv_path = tmp_path / "dataset.csv"
    csv_path.write_text(
        "dataset,vra1\ndataset,10\n",
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
