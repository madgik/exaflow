from __future__ import annotations

import logging
import sys
from pathlib import Path
from uuid import uuid4

import click as cl

from validator.exceptions import ExitCode
from validator.exceptions import InvalidDatasetError
from validator.exceptions import handle_errors
from validator.reporting import render_report
from validator.usecases import validate_data_model_folder

logging.basicConfig(
    stream=sys.stderr, level=logging.INFO, format="%(levelname)s: %(message)s"
)
LOGGER = logging.getLogger(__name__)


@cl.group()
def cli():
    """exadata-validator command line interface."""


def _format_report_link(output_path: Path) -> str:
    resolved = output_path.resolve()
    uri = resolved.as_uri()
    label = resolved.name
    if cl.get_text_stream("stdout").isatty():
        # OSC 8 hyperlink sequence for terminals that support clickable links.
        return f"\033]8;;{uri}\033\\{label}\033]8;;\033\\"
    return uri


def _default_html_report_path() -> Path:
    return Path("/tmp") / f"exadata-validator-report-{uuid4().hex}.html"


@cli.command("validate-data-model")
@cl.argument("folder", type=cl.Path(exists=True, file_okay=False, path_type=Path))
@cl.option(
    "--threads",
    type=cl.IntRange(min=1),
    default=None,
    help="DuckDB worker threads (defaults to CPU count).",
)
@cl.option(
    "--report-all/--fail-fast",
    default=True,
    help="Collect and report all validation errors instead of failing at the first one.",
)
@cl.option(
    "--format",
    "output_format",
    type=cl.Choice(["text", "html"], case_sensitive=False),
    default="text",
    show_default=True,
    help="Output format when collecting all validation errors.",
)
@cl.option(
    "--output",
    "output_path",
    type=cl.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help="Optional output file path when collecting all validation errors.",
)
@handle_errors
def validate_data_model(
    folder: Path,
    threads: int | None,
    report_all: bool,
    output_format: str,
    output_path: Path | None,
):
    report = validate_data_model_folder(folder, threads=threads, report_all=report_all)

    if report_all:
        selected_format = output_format.lower()
        rendered = render_report(report, selected_format)
        target_output_path = output_path
        if selected_format == "html" and target_output_path is None:
            target_output_path = _default_html_report_path()

        if target_output_path:
            target_output_path.write_text(rendered + "\n", encoding="utf-8")
            cl.echo(f"Report written to {_format_report_link(target_output_path)}")
        else:
            cl.echo(rendered)
        if report.has_errors:
            raise SystemExit(ExitCode.FILE_ERROR)
    elif report.has_errors:
        raise InvalidDatasetError(report.issues[0].message)

    LOGGER.info("Validation completed successfully for %s", folder)


entry = cli
