from __future__ import annotations

import copy
import difflib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import duckdb

from validator.dataelements import flatten_cdes
from validator.dataelements import get_cdes_with_enumerations
from validator.dataelements import get_cdes_with_min_max
from validator.dataelements import get_dataset_enums
from validator.dataelements import get_sql_type_per_column
from validator.exceptions import InvalidDatasetError
from validator.reporting import ValidationIssue


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _format_in_list(values: list[str]) -> str:
    return ", ".join(_quote_literal(value) for value in values)


def _format_allowed_values(values: list[str], max_items: int = 8) -> str:
    shown = values[:max_items]
    formatted = ", ".join(repr(value) for value in shown)
    if len(values) > max_items:
        return f"{formatted}, ... (total={len(values)})"
    return formatted


@dataclass(frozen=True)
class _FusedCheck:
    key: str
    rule: str
    column: str | None
    condition_sql: str
    sample_expression_sql: str
    message_factory: Callable[[int, str | None], str]


class DuckDBDatasetValidator:
    """Validate CSV files against a data model using direct DuckDB CSV queries."""

    def __init__(self, data_model_metadata: dict, threads: int | None = None) -> None:
        cdes_list = flatten_cdes(copy.deepcopy(data_model_metadata))
        self._cdes = {cde.code: cde for cde in cdes_list}
        self._sql_type_per_column = get_sql_type_per_column(self._cdes)
        self._dataset_enumerations = list(get_dataset_enums(self._cdes).keys())
        self._longitudinal = bool(data_model_metadata.get("longitudinal", False))
        self._threads = threads if threads is not None else (os.cpu_count() or 1)

    def validate_csv(
        self, csv_path: Path, report_all: bool = False
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        conn = duckdb.connect(database=":memory:")
        try:
            conn.execute(f"PRAGMA threads={self._threads}")

            columns = self._get_columns(conn, csv_path)
            issues.extend(self._validate_columns(columns, csv_path))

            if issues and not report_all:
                raise InvalidDatasetError(issues[0].message)

            fused_checks = self._build_fused_checks(columns)
            issues.extend(self._run_fused_checks(conn, csv_path, fused_checks))

            if self._longitudinal and {"subjectid", "visitid"} <= set(columns):
                issues.extend(self._validate_longitudinal_pairs(conn, csv_path))
        except duckdb.Error as exc:
            message = f"Unable to validate csv '{csv_path.name}'. {exc}"
            if report_all:
                issues.append(
                    ValidationIssue(
                        rule="csv.duckdb_error",
                        message=message,
                        file=str(csv_path),
                    )
                )
                return issues
            raise InvalidDatasetError(message) from exc
        finally:
            conn.close()

        if issues and not report_all:
            raise InvalidDatasetError(issues[0].message)
        return issues

    def _get_columns(
        self, conn: duckdb.DuckDBPyConnection, csv_path: Path
    ) -> list[str]:
        info = conn.execute(
            (
                "DESCRIBE SELECT * "
                "FROM read_csv_auto(?, header=true, all_varchar=true, nullstr=[''])"
            ),
            [str(csv_path)],
        ).fetchall()
        return [str(row[0]) for row in info]

    def _validate_columns(
        self, columns: list[str], csv_path: Path
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        if "dataset" not in columns:
            issues.append(
                ValidationIssue(
                    rule="columns.dataset_required",
                    message="The 'dataset' column is required to exist in the csv.",
                    file=str(csv_path),
                    column="dataset",
                )
            )

        unknown_columns = (
            set(columns) - set(self._sql_type_per_column.keys()) - {"row_id"}
        )
        for column in sorted(unknown_columns):
            suggestion = self._get_column_suggestion(column)
            message = f"Column '{column}' is not present in the CDEs."
            if suggestion:
                message += f" Did you mean '{suggestion}'?"
            issues.append(
                ValidationIssue(
                    rule="columns.unknown",
                    message=message,
                    file=str(csv_path),
                    column=column,
                )
            )

        if self._longitudinal:
            for required in ("subjectid", "visitid"):
                if required not in columns:
                    issues.append(
                        ValidationIssue(
                            rule="longitudinal.required_column",
                            message=(
                                "The "
                                f"'{required}' column is required for longitudinal data models."
                            ),
                            file=str(csv_path),
                            column=required,
                        )
                    )

        return issues

    def _get_column_suggestion(self, unknown_column: str) -> str | None:
        candidates = [
            column for column in self._sql_type_per_column.keys() if column != "row_id"
        ]
        matches = difflib.get_close_matches(unknown_column, candidates, n=1, cutoff=0.7)
        return matches[0] if matches else None

    def _build_fused_checks(self, columns: list[str]) -> list[_FusedCheck]:
        checks: list[_FusedCheck] = []

        type_mapping = {"int": "BIGINT", "real": "DOUBLE"}
        for column in columns:
            sql_type = self._sql_type_per_column.get(column)
            cast_type = type_mapping.get(sql_type)
            if not cast_type:
                continue

            quoted_column = _quote_identifier(column)
            checks.append(
                _FusedCheck(
                    key=f"type_{column}",
                    rule="types.invalid",
                    column=column,
                    condition_sql=(
                        f"{quoted_column} IS NOT NULL "
                        f"AND TRY_CAST({quoted_column} AS {cast_type}) IS NULL"
                    ),
                    sample_expression_sql=quoted_column,
                    message_factory=lambda count, sample, c=column, t=sql_type: (
                        f"Column '{c}' has invalid {t} values "
                        f"(count={count}, example={sample!r})."
                    ),
                )
            )

        cdes_with_min_max = get_cdes_with_min_max(self._cdes, columns)
        for column, (min_value, max_value) in cdes_with_min_max.items():
            quoted_column = _quote_identifier(column)
            if min_value is not None:
                checks.append(
                    _FusedCheck(
                        key=f"min_{column}",
                        rule="range.min",
                        column=column,
                        condition_sql=(
                            f"{quoted_column} IS NOT NULL "
                            f"AND TRY_CAST({quoted_column} AS DOUBLE) < {min_value}"
                        ),
                        sample_expression_sql=quoted_column,
                        message_factory=lambda count, sample, c=column, v=min_value: (
                            f"Column '{c}' has values below minimum {v} "
                            f"(count={count}, example={sample!r})."
                        ),
                    )
                )
            if max_value is not None:
                checks.append(
                    _FusedCheck(
                        key=f"max_{column}",
                        rule="range.max",
                        column=column,
                        condition_sql=(
                            f"{quoted_column} IS NOT NULL "
                            f"AND TRY_CAST({quoted_column} AS DOUBLE) > {max_value}"
                        ),
                        sample_expression_sql=quoted_column,
                        message_factory=lambda count, sample, c=column, v=max_value: (
                            f"Column '{c}' has values above maximum {v} "
                            f"(count={count}, example={sample!r})."
                        ),
                    )
                )

        cdes_with_enumerations = get_cdes_with_enumerations(self._cdes, columns)
        for column, allowed_values in cdes_with_enumerations.items():
            if not allowed_values:
                continue
            quoted_column = _quote_identifier(column)
            in_list = _format_in_list(allowed_values)
            allowed_display = _format_allowed_values(allowed_values)
            checks.append(
                _FusedCheck(
                    key=f"enum_{column}",
                    rule="enum.invalid",
                    column=column,
                    condition_sql=(
                        f"{quoted_column} IS NOT NULL AND {quoted_column} NOT IN ({in_list})"
                    ),
                    sample_expression_sql=quoted_column,
                    message_factory=lambda count, sample, c=column, allowed=allowed_display: (
                        f"Column '{c}' has invalid categorical value(s). "
                        f"Allowed values: [{allowed}]. "
                        f"Found {count} invalid row(s), example invalid value: {sample!r}."
                    ),
                )
            )

        if "dataset" in columns:
            in_list = _format_in_list(self._dataset_enumerations)
            dataset_allowed_display = _format_allowed_values(self._dataset_enumerations)
            checks.append(
                _FusedCheck(
                    key="dataset_enum",
                    rule="dataset.enum",
                    column="dataset",
                    condition_sql=f"dataset IS NOT NULL AND dataset NOT IN ({in_list})",
                    sample_expression_sql="dataset",
                    message_factory=lambda count, sample, allowed=dataset_allowed_display: (
                        "Column 'dataset' has value(s) not declared in CDEsMetadata "
                        f"enumerations. Allowed values: [{allowed}]. "
                        f"Found {count} invalid row(s), example invalid value: {sample!r}."
                    ),
                )
            )

        if self._longitudinal and "subjectid" in columns:
            checks.append(
                _FusedCheck(
                    key="longitudinal_subjectid_null",
                    rule="longitudinal.subjectid_null",
                    column="subjectid",
                    condition_sql="subjectid IS NULL",
                    sample_expression_sql=_quote_literal("NULL"),
                    message_factory=lambda count, _: (
                        "Column 'subjectid' should never contain null values "
                        f"(count={count})."
                    ),
                )
            )
        if self._longitudinal and "visitid" in columns:
            checks.append(
                _FusedCheck(
                    key="longitudinal_visitid_null",
                    rule="longitudinal.visitid_null",
                    column="visitid",
                    condition_sql="visitid IS NULL",
                    sample_expression_sql=_quote_literal("NULL"),
                    message_factory=lambda count, _: (
                        "Column 'visitid' should never contain null values "
                        f"(count={count})."
                    ),
                )
            )

        return checks

    def _run_fused_checks(
        self,
        conn: duckdb.DuckDBPyConnection,
        csv_path: Path,
        checks: list[_FusedCheck],
    ) -> list[ValidationIssue]:
        if not checks:
            return []

        projections = []
        for check in checks:
            count_alias = _quote_identifier(f"{check.key}__count")
            sample_alias = _quote_identifier(f"{check.key}__sample")
            row_alias = _quote_identifier(f"{check.key}__row")
            projections.append(
                f"SUM(CASE WHEN {check.condition_sql} THEN 1 ELSE 0 END) AS {count_alias}"
            )
            projections.append(
                (
                    "MIN(CASE WHEN "
                    f"{check.condition_sql} THEN {check.sample_expression_sql} "
                    f"ELSE NULL END) AS {sample_alias}"
                )
            )
            projections.append(
                f"MIN(CASE WHEN {check.condition_sql} THEN _rownum ELSE NULL END) AS {row_alias}"
            )

        query = (
            "WITH csv_data AS ("
            "  SELECT row_number() OVER () AS _rownum, * "
            "  FROM read_csv_auto(?, header=true, all_varchar=true, nullstr=[''])"
            ") "
            "SELECT " + ", ".join(projections) + " FROM csv_data"
        )
        row = conn.execute(query, [str(csv_path)]).fetchone()
        if row is None:
            return []

        issues: list[ValidationIssue] = []
        for index, check in enumerate(checks):
            count = int(row[index * 3] or 0)
            sample = row[index * 3 + 1]
            first_data_row = row[index * 3 + 2]
            if count <= 0:
                continue
            line_number = (
                int(first_data_row) + 1 if first_data_row is not None else None
            )
            message = check.message_factory(count, sample)
            if line_number is not None:
                message += f" First seen at line {line_number}."
            issues.append(
                ValidationIssue(
                    rule=check.rule,
                    message=message,
                    file=str(csv_path),
                    column=check.column,
                    row=line_number,
                )
            )
        return issues

    def _validate_longitudinal_pairs(
        self, conn: duckdb.DuckDBPyConnection, csv_path: Path
    ) -> list[ValidationIssue]:
        duplicates = conn.execute(
            (
                "SELECT subjectid, visitid, COUNT(*) AS duplicate_count "
                "FROM read_csv_auto(?, header=true, all_varchar=true, nullstr=['']) "
                "GROUP BY subjectid, visitid "
                "HAVING COUNT(*) > 1 "
                "LIMIT 5"
            ),
            [str(csv_path)],
        ).fetchall()
        if not duplicates:
            return []

        return [
            ValidationIssue(
                rule="longitudinal.duplicate_pair",
                message=(
                    "Invalid csv: duplicate (visitid, subjectid) pairs detected: "
                    f"{[(subject, visit) for subject, visit, _ in duplicates]}"
                ),
                file=str(csv_path),
            )
        ]
