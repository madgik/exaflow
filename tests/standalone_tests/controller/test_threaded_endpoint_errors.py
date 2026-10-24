import asyncio

import pytest

from exaflow.controller.quart import endpoints
from exaflow.controller.quart.app import app
from exaflow.controller.services.api.analysis_request_validator import BadRequest
from exaflow.controller.services.errors import WorkerTaskTimeoutError
from exaflow.controller.services.errors import WorkerUnresponsiveError
from exaflow.worker_communication import BadUserInput


async def _get(path):
    async with app.test_client() as client:
        response = await client.get(path)
        return response.status_code, await response.get_data(as_text=False)


async def _post(path):
    async with app.test_client() as client:
        response = await client.post(path)
        return response.status_code, await response.get_data(as_text=False)


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (WorkerUnresponsiveError(), 512),
        (WorkerTaskTimeoutError(), 513),
        (BadUserInput("missing column"), 460),
    ],
)
def test_healthcheck_errors_from_the_worker_thread_keep_their_status(
    monkeypatch, error, status
):
    def healthcheck():
        raise error

    monkeypatch.setattr(
        endpoints,
        "get_worker_landscape_aggregator",
        lambda: type("Wla", (), {"healthcheck": staticmethod(healthcheck)})(),
    )

    status_code, body = asyncio.run(_get("/healthcheck"))

    assert status_code == status
    assert error.message.encode() in body


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (BadRequest("landscape update rejected"), 400),
        (BadUserInput("unknown data model"), 460),
        (WorkerUnresponsiveError(), 512),
    ],
)
def test_wla_update_errors_from_the_worker_thread_keep_their_status(
    monkeypatch, error, status
):
    def update():
        raise error

    monkeypatch.setattr(
        endpoints,
        "get_worker_landscape_aggregator",
        lambda: type("Wla", (), {"update": staticmethod(update)})(),
    )

    status_code, body = asyncio.run(_post("/wla"))

    assert status_code == status
    assert error.message.encode() in body
