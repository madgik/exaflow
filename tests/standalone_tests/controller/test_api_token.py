"""The controller's API refuses callers without the token, once one is set."""

import asyncio
import importlib

import pytest

from exaflow.controller.quart.app import app


def status(path, headers=None):
    async def get():
        return await app.test_client().get(path, headers=headers or {})

    return asyncio.run(get()).status_code


def test_no_token_configured_keeps_the_api_open(monkeypatch):
    monkeypatch.delenv("EXAFLOW_API_TOKEN", raising=False)
    assert status("/livez") == 200
    assert status("/no-such-route") == 404


@pytest.mark.parametrize(
    "headers",
    [None, {"Authorization": "Bearer wrong"}, {"Authorization": "secret"}],
)
def test_wrong_or_missing_token_is_refused(monkeypatch, headers):
    monkeypatch.setenv("EXAFLOW_API_TOKEN", "secret")
    assert status("/no-such-route", headers) == 401


def test_right_token_passes(monkeypatch):
    monkeypatch.setenv("EXAFLOW_API_TOKEN", "secret")
    assert status("/no-such-route", {"Authorization": "Bearer secret"}) == 404


def test_probe_paths_need_no_token(monkeypatch):
    monkeypatch.setenv("EXAFLOW_API_TOKEN", "secret")
    assert status("/livez") == 200


def test_flower_sends_the_token(monkeypatch):
    flower = pytest.importorskip("exaflow.algorithms.flower.inputdata_preprocessing")
    monkeypatch.setenv("EXAFLOW_API_TOKEN", "secret")
    importlib.reload(flower)
    assert flower.AUTH_HEADERS == {"Authorization": "Bearer secret"}
    assert flower.HEADERS["Authorization"] == "Bearer secret"

    monkeypatch.delenv("EXAFLOW_API_TOKEN")
    importlib.reload(flower)
    assert flower.AUTH_HEADERS == {}
