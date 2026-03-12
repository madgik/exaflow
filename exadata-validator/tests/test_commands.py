import json
from pathlib import Path

from click.testing import CliRunner
from validator import commands
from validator.commands import entry as cli
from validator.exceptions import ExitCode

TEST_DIR = Path(__file__).resolve().parent
DATA_DIR = TEST_DIR / "data"


def run_cli(*args):
    runner = CliRunner()
    return runner.invoke(cli, [*args])


def test_validate_data_model_command_success():
    folder = DATA_DIR / "success" / "data_model1_v_1_0"
    result = run_cli("validate-data-model", str(folder))
    assert result.exit_code == ExitCode.OK


def test_validate_data_model_command_failure():
    folder = DATA_DIR / "fail" / "data_model_v_1_0"
    result = run_cli("validate-data-model", str(folder))
    assert result.exit_code == ExitCode.FILE_ERROR
    assert "Validation report for:" in result.output
    assert "[columns.unknown]" in result.output


def test_validate_data_model_command_fail_fast():
    folder = DATA_DIR / "fail" / "data_model_v_1_0"
    result = run_cli("validate-data-model", str(folder), "--fail-fast")
    assert result.exit_code == ExitCode.FILE_ERROR
    assert "Validation error" in result.output


def test_validate_data_model_command_no_csv(tmp_path):
    folder = tmp_path / "data_model"
    folder.mkdir()
    (folder / "CDEsMetadata.json").write_text(
        (DATA_DIR / "success" / "data_model_v_1_0" / "CDEsMetadata.json").read_text(),
        encoding="utf-8",
    )

    result = run_cli("validate-data-model", str(folder))
    assert result.exit_code == ExitCode.USER_ERROR
    assert "No CSV files found" in result.output


def test_validate_data_model_command_duplicate_dataset_across_csv_files(tmp_path):
    source_folder = DATA_DIR / "success" / "data_model_v_1_0"
    folder = tmp_path / "data_model"
    folder.mkdir()

    (folder / "CDEsMetadata.json").write_text(
        (source_folder / "CDEsMetadata.json").read_text(),
        encoding="utf-8",
    )
    (folder / "dataset1.csv").write_text(
        (source_folder / "dataset1.csv").read_text(),
        encoding="utf-8",
    )
    (folder / "dataset2.csv").write_text(
        (source_folder / "dataset2.csv").read_text().replace("dataset2", "dataset1"),
        encoding="utf-8",
    )

    result = run_cli("validate-data-model", str(folder))
    assert result.exit_code == ExitCode.FILE_ERROR
    assert "Dataset code collision after normalization" in result.output


def test_validate_data_model_command_rejects_unsupported_format():
    folder = DATA_DIR / "fail" / "data_model_v_1_0"
    result = run_cli("validate-data-model", str(folder), "--format", "ndjson")
    assert result.exit_code == 2
    assert "Invalid value for '--format'" in result.output


def test_validate_data_model_command_report_all_html_output_file(tmp_path):
    folder = DATA_DIR / "fail" / "data_model_v_1_0"
    output_path = tmp_path / "report.html"
    result = run_cli(
        "validate-data-model",
        str(folder),
        "--format",
        "html",
        "--output",
        str(output_path),
    )
    assert result.exit_code == ExitCode.FILE_ERROR
    assert output_path.exists()
    assert "<table>" in output_path.read_text(encoding="utf-8")


def test_validate_data_model_command_report_all_html_defaults_to_tmp(
    monkeypatch, tmp_path
):
    folder = DATA_DIR / "fail" / "data_model_v_1_0"
    auto_output_path = tmp_path / "report-auto.html"
    monkeypatch.setattr(commands, "_default_html_report_path", lambda: auto_output_path)

    result = run_cli("validate-data-model", str(folder), "--format", "html")

    assert result.exit_code == ExitCode.FILE_ERROR
    assert auto_output_path.exists()
    assert "<table>" in auto_output_path.read_text(encoding="utf-8")
    assert "Report written to" in result.output


def test_validate_data_model_command_duplicate_dataset_normalization(tmp_path):
    source_folder = DATA_DIR / "success" / "data_model_v_1_0"
    folder = tmp_path / "data_model"
    folder.mkdir()

    metadata = json.loads(
        (source_folder / "CDEsMetadata.json").read_text(encoding="utf-8")
    )
    dataset_cde = next(cde for cde in metadata["variables"] if cde["code"] == "dataset")
    dataset_cde["enumerations"].append({"code": "DATASET1", "label": "Dataset 1 Upper"})
    (folder / "CDEsMetadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    (folder / "dataset1.csv").write_text(
        (source_folder / "dataset1.csv").read_text(),
        encoding="utf-8",
    )
    (folder / "dataset2.csv").write_text(
        (source_folder / "dataset2.csv").read_text().replace("dataset2", "DATASET1"),
        encoding="utf-8",
    )

    result = run_cli("validate-data-model", str(folder))
    assert result.exit_code == ExitCode.FILE_ERROR
    assert "collision after normalization (trim+lower)" in result.output


def test_validate_data_model_command_report_all_text_includes_line_number(tmp_path):
    source_folder = DATA_DIR / "success" / "data_model_v_1_0"
    folder = tmp_path / "data_model"
    folder.mkdir()

    (folder / "CDEsMetadata.json").write_text(
        (source_folder / "CDEsMetadata.json").read_text(),
        encoding="utf-8",
    )
    (folder / "dataset.csv").write_text(
        "subjectcode,var1,var3,dataset\n10,1,2000,dataset1\n",
        encoding="utf-8",
    )

    result = run_cli("validate-data-model", str(folder), "--format", "text")
    assert result.exit_code == ExitCode.FILE_ERROR
    assert "[range.max]" in result.output
    assert "line=2" in result.output
