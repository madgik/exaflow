from __future__ import annotations

import copy
import json
from pathlib import Path

import duckdb

from validator.dataelements import flatten_cdes
from validator.dataelements import validate_dataset_present_on_cdes_with_proper_format
from validator.dataelements import validate_longitudinal_data_model
from validator.duckdb_validator import DuckDBDatasetValidator
from validator.exceptions import FileContentError
from validator.exceptions import InvalidDataModelError
from validator.exceptions import InvalidDatasetError
from validator.exceptions import UserInputError
from validator.reporting import ValidationIssue
from validator.reporting import ValidationReport

LONGITUDINAL = "longitudinal"


def _quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _read_json_file(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        try:
            return json.load(stream)
        except json.JSONDecodeError as exc:
            raise FileContentError(
                f"Unable to decode json file. {exc.args[0]}"
            ) from exc


def validate_data_model_metadata(data_model_metadata: dict) -> None:
    if "version" not in data_model_metadata:
        raise UserInputError("You need to include a version on the CDEsMetadata.json")

    cdes = flatten_cdes(copy.deepcopy(data_model_metadata))
    validate_dataset_present_on_cdes_with_proper_format(cdes)

    if LONGITUDINAL in data_model_metadata:
        longitudinal = data_model_metadata[LONGITUDINAL]
        if not isinstance(longitudinal, bool):
            raise UserInputError(
                f"Longitudinal flag should be boolean, value given: {longitudinal}"
            )
        if longitudinal:
            validate_longitudinal_data_model(cdes)


def _validate_dataset_uniqueness_with_sql(
    csv_files: list[Path],
) -> list[ValidationIssue]:
    if len(csv_files) < 2:
        return []

    file_list_sql = ", ".join(_quote_literal(str(path)) for path in csv_files)
    query = (
        "WITH csv_rows AS ("
        "  SELECT filename, dataset "
        f"  FROM read_csv_auto([{file_list_sql}], "
        "       header=true, all_varchar=true, nullstr=[''], "
        "       filename=true, union_by_name=true)"
        "), normalized AS ("
        "  SELECT filename, NULLIF(LOWER(TRIM(dataset)), '') AS dataset_normalized "
        "  FROM csv_rows "
        "  WHERE dataset IS NOT NULL"
        ") "
        "SELECT dataset_normalized, list(DISTINCT filename) AS files "
        "FROM normalized "
        "WHERE dataset_normalized IS NOT NULL "
        "GROUP BY dataset_normalized "
        "HAVING COUNT(DISTINCT filename) > 1 "
        "ORDER BY dataset_normalized"
    )

    conn = duckdb.connect(database=":memory:")
    try:
        rows = conn.execute(query).fetchall()
    except duckdb.Error as exc:
        raise InvalidDatasetError(
            f"Unable to validate folder-level dataset uniqueness: {exc}"
        ) from exc
    finally:
        conn.close()

    issues: list[ValidationIssue] = []
    for dataset_normalized, files in rows:
        file_names = sorted(Path(str(path)).name for path in files)
        issues.append(
            ValidationIssue(
                rule="folder.dataset_uniqueness",
                message=(
                    "Dataset code collision after normalization (trim+lower) for "
                    f"'{dataset_normalized}' across files: {file_names}"
                ),
            )
        )
    return issues


def validate_data_model_folder(
    folder: Path, *, threads: int | None = None, report_all: bool = False
) -> ValidationReport:
    report = ValidationReport(folder=str(folder))

    metadata_path = folder / "CDEsMetadata.json"
    if not metadata_path.exists():
        raise UserInputError(f"Missing metadata file: {metadata_path}")

    data_model_metadata = _read_json_file(metadata_path)
    try:
        validate_data_model_metadata(data_model_metadata)
    except (InvalidDataModelError, UserInputError) as exc:
        if not report_all:
            raise
        report.issues.append(
            ValidationIssue(
                rule="metadata.invalid",
                message=str(exc),
                file=str(metadata_path),
            )
        )
        return report

    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise UserInputError(
            f"No CSV files found in {folder}. Expected at least one dataset csv."
        )
    report.files_checked = [str(csv_file) for csv_file in csv_files]

    validator = DuckDBDatasetValidator(data_model_metadata, threads=threads)
    for csv_file in csv_files:
        csv_issues = validator.validate_csv(csv_file, report_all=report_all)
        report.issues.extend(csv_issues)

    try:
        duplicate_issues = _validate_dataset_uniqueness_with_sql(csv_files)
        report.issues.extend(duplicate_issues)
    except InvalidDatasetError as exc:
        if not report_all:
            raise
        report.issues.append(
            ValidationIssue(
                rule="folder.dataset_uniqueness",
                message=str(exc),
            )
        )

    if report.issues and not report_all:
        raise InvalidDatasetError(report.issues[0].message)

    return report
