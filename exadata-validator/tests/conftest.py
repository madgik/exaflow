import json
from pathlib import Path

import pytest

TEST_DIR = Path(__file__).resolve().parent
DATA_DIR = TEST_DIR / "data"

SUCCESS_MODEL_DIR = DATA_DIR / "success" / "data_model_v_1_0"
FAIL_MODEL_DIR = DATA_DIR / "fail" / "data_model_v_1_0"

SUCCESS_METADATA_PATH = SUCCESS_MODEL_DIR / "CDEsMetadata.json"
FAIL_METADATA_PATH = FAIL_MODEL_DIR / "CDEsMetadata.json"

LONGITUDINAL_SUCCESS_DIR = DATA_DIR / "success" / "data_model_longitudinal_v_1_0"
LONGITUDINAL_FAIL_DIR = DATA_DIR / "fail" / "data_model_longitudinal_v_1_0"


@pytest.fixture
def success_metadata():
    return json.loads(SUCCESS_METADATA_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def fail_metadata():
    return json.loads(FAIL_METADATA_PATH.read_text(encoding="utf-8"))
