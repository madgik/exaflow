from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
from html import escape
from pathlib import Path


@dataclass
class ValidationIssue:
    rule: str
    message: str
    file: str | None = None
    column: str | None = None
    row: int | None = None
    severity: str = "error"

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "rule": self.rule,
            "message": self.message,
            "file": self.file,
            "column": self.column,
            "row": self.row,
        }


@dataclass
class ValidationReport:
    folder: str
    files_checked: list[str] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def has_errors(self) -> bool:
        return bool(self.issues)

    def to_dict(self) -> dict:
        issues_by_file: dict[str, list[dict]] = {}
        for file_key, grouped_issues in _group_issues_by_file(self):
            key = file_key if file_key is not None else "__folder__"
            issues_by_file[key] = [issue.to_dict() for issue in grouped_issues]

        return {
            "folder": self.folder,
            "generated_at": self.generated_at,
            "files_checked": self.files_checked,
            "error_count": len(self.issues),
            "issues": [issue.to_dict() for issue in self.issues],
            "issues_by_file": issues_by_file,
        }


def render_report(report: ValidationReport, output_format: str) -> str:
    if output_format == "text":
        return _render_text(report)
    if output_format == "html":
        return _render_html(report)
    raise ValueError(f"Unsupported report format: {output_format}")


def _group_issues_by_file(
    report: ValidationReport,
) -> list[tuple[str | None, list[ValidationIssue]]]:
    grouped: dict[str | None, list[ValidationIssue]] = {}
    for issue in report.issues:
        grouped.setdefault(issue.file, []).append(issue)

    ordered: list[tuple[str | None, list[ValidationIssue]]] = []
    for file_path in report.files_checked:
        issues = grouped.pop(file_path, None)
        if issues:
            ordered.append((file_path, issues))

    for file_path, issues in grouped.items():
        ordered.append((file_path, issues))

    return ordered


def _render_text(report: ValidationReport) -> str:
    lines = [
        f"Validation report for: {report.folder}",
        f"Files checked: {len(report.files_checked)}",
        f"Errors: {len(report.issues)}",
    ]

    grouped_issues = _group_issues_by_file(report)
    if not grouped_issues:
        lines.append("No validation errors were found.")
        return "\n".join(lines)

    index = 1
    for file_path, issues in grouped_issues:
        if file_path is None:
            lines.append("Folder-level issues:")
        else:
            lines.append(f"CSV: {file_path}")
        for issue in issues:
            location_parts = []
            if issue.column:
                location_parts.append(f"column={issue.column}")
            if issue.row is not None:
                location_parts.append(f"line={issue.row}")
            location = ", ".join(location_parts) if location_parts else "global"
            lines.append(f"{index}. [{issue.rule}] {location}: {issue.message}")
            index += 1
    return "\n".join(lines)


def _render_html(report: ValidationReport) -> str:
    grouped_sections: list[str] = []
    grouped_issues = _group_issues_by_file(report)
    for file_path, issues in grouped_issues:
        section_title = "Folder-level issues"
        title_tooltip = "Folder-level validation issues"
        if file_path is not None:
            file_name = Path(file_path).name
            section_title = file_name
            title_tooltip = file_path

        rows = []
        for issue in issues:
            severity_class = f"severity-{escape(issue.severity).lower()}"
            rows.append(
                "<tr>"
                f"<td><span class='severity {severity_class}'>{escape(issue.severity)}</span></td>"
                f"<td>{escape(issue.rule)}</td>"
                f"<td>{escape(issue.column or '')}</td>"
                f"<td>{'' if issue.row is None else issue.row}</td>"
                f"<td>{escape(issue.message)}</td>"
                "</tr>"
            )

        table_rows = "\n".join(rows)
        grouped_sections.append(
            "<section class='file-section'>"
            "<div class='file-header'>"
            f"<h2 class='file-title' title='{escape(title_tooltip)}'>{escape(section_title)}</h2>"
            f"<p class='file-count'>{len(issues)} error(s)</p>"
            "</div>"
            "<div class='table-wrap'>"
            "<table>"
            "<colgroup>"
            "<col style='width:90px'>"
            "<col style='width:150px'>"
            "<col style='width:22%'>"
            "<col style='width:70px'>"
            "<col>"
            "</colgroup>"
            "<thead><tr>"
            "<th>Severity</th><th>Rule</th><th>Column</th><th>Line</th><th>Message</th>"
            "</tr></thead><tbody>"
            f"{table_rows}"
            "</tbody></table></div></section>"
        )

    sections_html = (
        "".join(grouped_sections)
        if grouped_sections
        else (
            "<div class='table-wrap'><table><thead><tr>"
            "<th>Severity</th><th>Rule</th><th>Column</th><th>Line</th><th>Message</th>"
            "</tr></thead><tbody><tr><td colspan='5'><div class='empty-state'>"
            "No validation errors were found."
            "</div></td></tr></tbody></table></div>"
        )
    )
    return (
        "<!doctype html>"
        "<html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Data Validation Report</title>"
        "<style>"
        ":root{"
        "--bg:#f4f7fb;"
        "--card:#ffffff;"
        "--text:#12263a;"
        "--muted:#5a6b7c;"
        "--border:#d7e0ea;"
        "--header:#e8eef6;"
        "--accent:#0c6d9a;"
        "--error-bg:#ffe9e8;"
        "--error-text:#8f1e18;"
        "}"
        "body{"
        "margin:0;"
        "font-family:'Avenir Next','Trebuchet MS','Gill Sans',sans-serif;"
        "background:linear-gradient(160deg,#f7fbff 0%,#eef4fa 100%);"
        "color:var(--text);"
        "}"
        ".container{"
        "max-width:1120px;"
        "margin:32px auto;"
        "padding:0 20px;"
        "}"
        ".card{"
        "background:var(--card);"
        "border:1px solid var(--border);"
        "border-radius:16px;"
        "box-shadow:0 12px 30px rgba(8,39,64,.08);"
        "overflow:hidden;"
        "}"
        ".hero{"
        "padding:24px 28px 18px 28px;"
        "background:linear-gradient(135deg,#edf4fb 0%,#e4edf7 100%);"
        "border-bottom:1px solid var(--border);"
        "}"
        ".title{"
        "margin:0;"
        "font-size:28px;"
        "letter-spacing:.3px;"
        "}"
        ".subtitle{"
        "margin:6px 0 0 0;"
        "color:var(--muted);"
        "font-size:14px;"
        "}"
        ".summary{"
        "display:grid;"
        "grid-template-columns:repeat(auto-fit,minmax(180px,1fr));"
        "gap:12px;"
        "padding:18px 24px 22px 24px;"
        "}"
        ".stat{"
        "background:#f8fbff;"
        "border:1px solid var(--border);"
        "border-radius:12px;"
        "padding:12px 14px;"
        "}"
        ".stat-label{"
        "margin:0 0 6px 0;"
        "font-size:12px;"
        "text-transform:uppercase;"
        "letter-spacing:.08em;"
        "color:var(--muted);"
        "}"
        ".stat-value{"
        "margin:0;"
        "font-size:16px;"
        "font-weight:600;"
        "line-height:1.3;"
        "word-break:break-word;"
        "}"
        ".file-section{margin:0 16px 18px 16px;border:1px solid var(--border);border-radius:12px;overflow:hidden;background:#fff;}"
        ".file-header{display:flex;justify-content:space-between;gap:12px;align-items:center;padding:12px 14px;background:#f3f8fe;border-bottom:1px solid var(--border);}"
        ".file-title{margin:0;font-size:16px;font-weight:700;cursor:help;word-break:break-word;}"
        ".file-count{margin:0;font-size:12px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.06em;}"
        ".table-wrap{padding:0;overflow-x:auto;}"
        "table{border-collapse:collapse;width:100%;min-width:920px;background:#fff;table-layout:fixed;}"
        "th,td{padding:10px 12px;text-align:left;vertical-align:top;border-bottom:1px solid var(--border);}"
        "th{"
        "background:var(--header);"
        "font-size:12px;"
        "text-transform:uppercase;"
        "letter-spacing:.06em;"
        "color:#27445d;"
        "position:sticky;"
        "top:0;"
        "}"
        "tbody tr:nth-child(even){background:#fbfdff;}"
        "tbody tr:hover{background:#f2f8fe;}"
        ".severity{"
        "display:inline-block;"
        "padding:3px 8px;"
        "border-radius:999px;"
        "font-size:11px;"
        "font-weight:700;"
        "text-transform:uppercase;"
        "letter-spacing:.05em;"
        "}"
        ".severity-error{background:var(--error-bg);color:var(--error-text);}"
        ".empty-state{padding:18px 0;color:var(--muted);font-style:italic;}"
        "@media (max-width:700px){"
        ".container{margin:18px auto;padding:0 10px;}"
        ".hero{padding:18px 16px;}"
        ".summary{padding:12px 14px 16px 14px;}"
        ".title{font-size:22px;}"
        "}"
        "</style></head><body>"
        "<div class='container'><section class='card'>"
        "<header class='hero'>"
        "<h1 class='title'>Data Validation Report</h1>"
        "<p class='subtitle'>Generated from exadata-validator</p>"
        "</header>"
        "<section class='summary'>"
        "<article class='stat'>"
        "<p class='stat-label'>Folder</p>"
        f"<p class='stat-value'>{escape(report.folder)}</p>"
        "</article>"
        "<article class='stat'>"
        "<p class='stat-label'>Generated</p>"
        f"<p class='stat-value'>{escape(report.generated_at)}</p>"
        "</article>"
        "<article class='stat'>"
        "<p class='stat-label'>Files Checked</p>"
        f"<p class='stat-value'>{len(report.files_checked)}</p>"
        "</article>"
        "<article class='stat'>"
        "<p class='stat-label'>Errors</p>"
        f"<p class='stat-value'>{len(report.issues)}</p>"
        "</article>"
        "</section>"
        f"{sections_html}"
        "</section></div></body></html>"
    )
