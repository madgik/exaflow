import asyncio

from exaflow.controller.quart import endpoints
from exaflow.controller.quart.app import app


def test_livez_does_not_contact_workers(monkeypatch):
    def _fail():
        raise AssertionError("liveness must not query the worker landscape")

    monkeypatch.setattr(endpoints, "get_worker_landscape_aggregator", _fail)

    async def _get():
        return await app.test_client().get("/livez")

    response = asyncio.run(_get())

    assert response.status_code == 200
