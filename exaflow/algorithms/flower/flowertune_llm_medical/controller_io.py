"""Controller I/O helpers for flowertune_llm_medical."""

from __future__ import annotations

import os
from typing import Any
from typing import Dict

import requests


CONTROLLER_IP = os.getenv("CONTROLLER_IP", "127.0.0.1")
CONTROLLER_PORT = os.getenv("CONTROLLER_PORT", "5000")
BASE_URL = f"http://{CONTROLLER_IP}:{CONTROLLER_PORT}"
INPUT_URL = f"{BASE_URL}/flower/input"
PARAMETERS_URL = f"{BASE_URL}/flower/parameters"
RUN_ENV_URL = f"{BASE_URL}/flower/run_env"
EVENT_URL = f"{BASE_URL}/flower/event"
RESULT_URL = f"{BASE_URL}/flower/result"
HEADERS = {"Content-type": "application/json", "Accept": "application/json"}


def _get_json(url: str) -> Dict[str, Any]:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def get_inputdata() -> Dict[str, Any]:
    request_id = os.getenv("REQUEST_ID")
    url = INPUT_URL if not request_id else f"{INPUT_URL}?request_id={request_id}"
    return _get_json(url)


def get_parameters() -> Dict[str, Any]:
    request_id = os.getenv("REQUEST_ID")
    url = PARAMETERS_URL if not request_id else f"{PARAMETERS_URL}?request_id={request_id}"
    return _get_json(url)


def get_run_env() -> Dict[str, str]:
    request_id = os.getenv("REQUEST_ID")
    url = RUN_ENV_URL if not request_id else f"{RUN_ENV_URL}?request_id={request_id}"
    data = _get_json(url)
    return {str(k): str(v) for k, v in data.items()}


def post_result(result: Dict[str, Any]) -> None:
    response = requests.post(RESULT_URL, json=result, headers=HEADERS, timeout=30)
    response.raise_for_status()


def post_event(event: Dict[str, Any]) -> None:
    request_id = os.getenv("REQUEST_ID")
    url = EVENT_URL if not request_id else f"{EVENT_URL}?request_id={request_id}"
    response = requests.post(url, json=event, headers=HEADERS, timeout=30)
    response.raise_for_status()
