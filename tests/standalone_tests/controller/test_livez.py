import asyncio

from exaflow.controller.http import endpoints
from exaflow.controller.http.app import app
from tests.standalone_tests.controller.asgi_client import controller_client


def test_livez_does_not_contact_workers(monkeypatch):
    def _fail():
        raise AssertionError("liveness must not query the worker landscape")

    monkeypatch.setattr(endpoints, "get_worker_landscape_aggregator", _fail)

    async def _get():
        async with controller_client(app) as client:
            return await client.get("/livez")

    response = asyncio.run(_get())

    assert response.status_code == 200
