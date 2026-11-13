"""Algorithm discovery does not depend on the process working directory.

The default folders are resolved from the installed package, and a folder
that does not exist is an error instead of an empty algorithm catalogue.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import exaflow

REPO_ROOT = Path(exaflow.__file__).resolve().parents[1]
FOLDER_ENV_VARIABLES = ("EXAREME3_ALGORITHM_FOLDERS", "FLOWER_ALGORITHM_FOLDERS")
REPORT_CODE = """
import json
import exaflow

print(json.dumps({
    "algorithms": sorted(exaflow.exareme3_algorithm_classes),
    "steps": sorted(exaflow.exareme3_preprocessing_step_classes),
    "exareme3_folders": exaflow.EXAREME3_ALGORITHM_FOLDERS,
    "flower_folders": exaflow.FLOWER_ALGORITHM_FOLDERS,
}))
"""


def _discover_from(working_directory: Path) -> dict:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in FOLDER_ENV_VARIABLES
    }
    env["PYTHONPATH"] = str(REPO_ROOT)
    output = subprocess.check_output(
        [sys.executable, "-c", REPORT_CODE],
        cwd=working_directory,
        env=env,
        text=True,
    )
    return json.loads(output.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def discovered_elsewhere(tmp_path_factory):
    return _discover_from(tmp_path_factory.mktemp("other_working_directory"))


def test_algorithms_are_discovered_from_another_working_directory(
    discovered_elsewhere,
):
    assert "describe" in discovered_elsewhere["algorithms"]
    assert "missing_values_handler" in discovered_elsewhere["steps"]


def test_default_folders_point_into_the_package(discovered_elsewhere):
    package_algorithms = REPO_ROOT / "exaflow" / "algorithms"

    assert discovered_elsewhere["exareme3_folders"] == str(
        package_algorithms / "exareme3"
    )
    assert discovered_elsewhere["flower_folders"] == str(package_algorithms / "flower")


def test_missing_algorithm_folder_is_an_error(tmp_path):
    missing = tmp_path / "missing_folder"

    with pytest.raises(FileNotFoundError, match="missing_folder"):
        exaflow.import_algorithm_modules(str(missing))


def test_folder_list_ignores_spaces_and_empty_entries(tmp_path):
    (tmp_path / "plain_module.py").write_text("VALUE = 1\n", encoding="utf-8")

    modules = exaflow.import_algorithm_modules(f" {tmp_path} , ")

    assert modules["plain_module"].VALUE == 1
