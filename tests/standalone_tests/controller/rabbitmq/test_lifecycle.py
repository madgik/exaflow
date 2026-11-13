import asyncio
from types import SimpleNamespace

from exaflow import AttrDict
from exaflow.controller.http import app as app_module
from exaflow.controller.services.rabbitmq.consumer import AnalysisConsumer
from exaflow.controller.services.rabbitmq.settings import RabbitMQSettings
from tests.standalone_tests.controller.asgi_client import controller_client


def test_lifespan_stops_consumer_before_controller_shutdown_and_always_closes(
    monkeypatch,
):
    events = []

    class Consumer:
        def __init__(self, settings, readiness):
            assert not settings.enabled

        def start(self):
            events.append("start-consumer")

        async def drain(self):
            events.append("drain-consumer")

        async def close(self):
            events.append("close-consumer")

    async def startup():
        events.append("start-controller")

    async def shutdown():
        events.append("stop-controller")
        raise RuntimeError("shutdown failed")

    monkeypatch.setattr(app_module, "ctrl_config", AttrDict({}))
    monkeypatch.setattr(app_module, "AnalysisConsumer", Consumer)
    monkeypatch.setattr(app_module.endpoints, "startup", startup)
    monkeypatch.setattr(app_module.endpoints, "shutdown", shutdown)

    async def scenario():
        import pytest

        with pytest.raises(RuntimeError, match="shutdown failed"):
            async with app_module.lifespan(app_module.app):
                pass

    previous = getattr(app_module.app.state, "rabbitmq", None)
    try:
        asyncio.run(scenario())
    finally:
        app_module.app.state.rabbitmq = previous
    assert events == [
        "start-controller",
        "start-consumer",
        "drain-consumer",
        "stop-controller",
        "close-consumer",
    ]


def test_broker_health_does_not_fail_http_liveness():
    async def scenario():
        runtime = AnalysisConsumer(
            RabbitMQSettings(enabled=True, url="amqp://localhost"), lambda: False
        )
        previous = getattr(app_module.app.state, "rabbitmq", None)
        app_module.app.state.rabbitmq = runtime
        try:
            async with controller_client(app_module.app) as client:
                assert (await client.get("/livez")).status_code == 200
                response = await client.get("/healthcheck/rabbitmq")
                assert response.status_code == 503
                assert response.json()["active_jobs"] == 0
                runtime.connection = SimpleNamespace(is_closed=False)
                runtime.consuming = True
                assert (await client.get("/healthcheck/rabbitmq")).status_code == 200
        finally:
            app_module.app.state.rabbitmq = previous

    asyncio.run(scenario())
