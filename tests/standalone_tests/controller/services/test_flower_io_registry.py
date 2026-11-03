"""Flower callbacks are accepted only for the request that is running."""

import ast
import asyncio
import json
import logging
from pathlib import Path

from exaflow.controller.quart import endpoints
from exaflow.controller.quart.app import app
from exaflow.controller.services.flower.flower_io_registry import FlowerIORegistry
from exaflow.controller.services.flower.flower_io_registry import Status

REPO_ROOT = Path(__file__).resolve().parents[4]
MISMATCH = "Flower callback does not belong to the running analysis."


def _registry() -> FlowerIORegistry:
    return FlowerIORegistry(timeout=5, logger=logging.getLogger("flower-io-test"))


def test_is_current_matches_only_the_running_request():
    registry = _registry()

    assert registry.is_current(None) is False
    assert registry.is_current("abc") is False

    registry.set_inputdata({"y": ["x"]}, "abc")

    assert registry.is_current("abc") is True
    assert registry.is_current("old") is False
    assert registry.is_current(None) is False
    assert registry.get_inputdata() == {"y": ["x"]}


def test_reset_clears_the_running_request_id():
    registry = _registry()
    registry.set_inputdata({"y": ["x"]}, "abc")

    asyncio.run(registry.reset())

    assert registry.is_current("abc") is False
    assert registry.is_current(None) is False


def test_set_result_status_still_follows_the_error_key():
    registry = _registry()
    registry.set_inputdata({}, "abc")

    asyncio.run(registry.set_result({"value": 1}))
    assert registry.get_status() == Status.SUCCESS

    registry.set_inputdata({}, "abc")
    asyncio.run(registry.set_result({"error": "failed"}))
    assert registry.get_status() == Status.FAILURE


def test_only_the_running_request_can_complete_the_execution(monkeypatch):
    registry = _registry()
    registry.set_inputdata({"inputdata": "value"}, "abc")
    monkeypatch.setattr(
        endpoints,
        "get_flower_controller",
        lambda: type("Controller", (), {"flower_execution_info": registry})(),
    )

    async def scenario():
        async with app.test_client() as client:
            stale = await client.post(
                "/flower/result",
                json={"late": True},
                headers={"X-Request-Id": "old"},
            )
            missing = await client.post("/flower/result", json={"late": True})
            waiting = not registry.result_ready.is_set()
            ok = await client.post(
                "/flower/result",
                json={"value": 1},
                headers={"X-Request-Id": "abc"},
            )
            got = await client.get("/flower/input", headers={"X-Request-Id": "abc"})
            rejected = await client.get(
                "/flower/input", headers={"X-Request-Id": "old"}
            )
            return {
                "stale": stale.status_code,
                "stale_body": await stale.get_data(as_text=True),
                "missing": missing.status_code,
                "waiting": waiting,
                "ok": ok.status_code,
                "got": await got.get_data(as_text=True),
                "rejected": rejected.status_code,
            }

    outcome = asyncio.run(scenario())

    assert outcome["waiting"] is True
    assert outcome["stale"] == 409
    assert outcome["stale_body"] == MISMATCH
    assert outcome["missing"] == 409
    assert outcome["ok"] == 200
    assert json.loads(outcome["got"]) == {"inputdata": "value"}
    assert outcome["rejected"] == 409
    assert registry.result_ready.is_set()
    assert registry.get_status() == Status.SUCCESS


def test_result_after_reset_is_ignored(monkeypatch):
    registry = _registry()
    registry.set_inputdata({"inputdata": "value"}, "abc")
    asyncio.run(registry.reset())
    monkeypatch.setattr(
        endpoints,
        "get_flower_controller",
        lambda: type("Controller", (), {"flower_execution_info": registry})(),
    )

    async def scenario():
        async with app.test_client() as client:
            response = await client.post(
                "/flower/result",
                json={"value": 1},
                headers={"X-Request-Id": "abc"},
            )
            return response.status_code, await response.get_data(as_text=True)

    status, body = asyncio.run(scenario())

    assert status == 409
    assert body == MISMATCH
    assert not registry.result_ready.is_set()


def test_flower_script_sends_request_id_and_timeout_on_controller_calls():
    script = "exaflow/algorithms/flower/inputdata_preprocessing.py"
    path = REPO_ROOT / script
    tree = ast.parse(path.read_text())

    request_id_assign = _assign(tree, "REQUEST_ID")
    getenv = request_id_assign.value
    assert isinstance(getenv, ast.Call)
    assert isinstance(getenv.func, ast.Attribute)
    assert getenv.func.value.id == "os"
    assert getenv.func.attr == "getenv"
    assert getenv.args[0].value == "REQUEST_ID"

    headers = _assign(tree, "HEADERS").value
    assert isinstance(headers, ast.Dict)
    header_keys = [key.value for key in headers.keys]
    request_header = headers.values[header_keys.index("X-Request-Id")]
    assert isinstance(request_header, ast.Name)
    assert request_header.id == "REQUEST_ID"

    calls = _requests_calls(tree)
    assert [(_url_name(call), _timeout(call)) for call in calls] == [
        ("RESULT_URL", 30),
        ("RESULT_URL", 30),
        ("INPUT_URL", 30),
        ("CDES_URL", 30),
    ]
    for call in calls:
        if _url_name(call) == "CDES_URL":
            continue
        headers_arg = next(kw.value for kw in call.keywords if kw.arg == "headers")
        assert isinstance(headers_arg, ast.Name)
        assert headers_arg.id == "HEADERS"


def _assign(tree: ast.AST, name: str) -> ast.Assign:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return node
    raise AssertionError(f"{name} is not assigned")


def _requests_calls(tree: ast.AST) -> list[ast.Call]:
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = node.func.value
        if isinstance(owner, ast.Name) and owner.id == "requests":
            calls.append(node)
    return calls


def _url_name(call: ast.Call) -> str:
    url = call.args[0]
    assert isinstance(url, ast.Name)
    return url.id


def _timeout(call: ast.Call) -> int:
    timeout = next(kw.value for kw in call.keywords if kw.arg == "timeout")
    assert isinstance(timeout, ast.Constant)
    return timeout.value
