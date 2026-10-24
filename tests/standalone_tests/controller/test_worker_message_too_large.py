import asyncio
import logging

import pytest

from exaflow import AttrDict
from exaflow.controller import DeploymentType
from exaflow.controller.services.flower.strategies import FlowerStrategy
from exaflow.controller.services.worker_landscape_aggregator import (
    worker_landscape_aggregator as wla_module,
)
from exaflow.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    WorkerLandscapeAggregator,
)


@pytest.fixture
def worker_landscape_aggregator():
    return WorkerLandscapeAggregator(
        logger=logging.getLogger("worker-landscape-tests"),
        update_interval=0,
        tasks_timeout=0,
        deployment_type=DeploymentType.LOCAL,
        localworkers=AttrDict({}),
    )


def _message_too_large():
    try:
        from exaflow.controller.worker_client.app import WorkerMessageTooLargeError
    except ImportError:
        pytest.fail(
            "WorkerMessageTooLargeError is not defined; "
            "RESOURCE_EXHAUSTED is still a generic connection error"
        )
    return WorkerMessageTooLargeError(
        connection_address="10.0.0.8:5672",
        error_details="Sent message larger than max (2097152 vs. 1048576)",
        max_message_bytes=1048576,
    )


def test_analysis_maps_worker_message_too_large(monkeypatch):
    from exaflow.controller.quart import endpoints
    from exaflow.controller.quart.app import app

    error = _message_too_large()

    class _Strategy:
        async def execute(self):
            raise error

    monkeypatch.setattr(endpoints, "get_worker_landscape_aggregator", lambda: None)
    monkeypatch.setattr(endpoints, "validate_analysis_request", lambda **kwargs: None)
    monkeypatch.setattr(
        endpoints, "get_algorithm_execution_strategy", lambda dto: _Strategy()
    )
    # Production does not propagate; force the registered handler to run.
    monkeypatch.setitem(app.config, "PROPAGATE_EXCEPTIONS", False)

    async def _post():
        client = app.test_client()
        response = await client.post(
            "/analysis",
            json={
                "inputdata": {
                    "data_model": "dementia",
                    "datasets": ["ppmi0"],
                    "variables": ["x"],
                },
                "algorithm": {"name": "pca", "y": ["x"]},
            },
        )
        body = await response.get_data(as_text=True)
        return response.status_code, body

    status, body = asyncio.run(_post())
    assert status == 413
    lowered = body.lower()
    assert "too large" in lowered
    assert "reduce variables" in lowered
    assert "1048576" in body or "1 mib" in lowered
    assert "connection error" not in lowered


def test_wla_does_not_drop_worker_for_message_too_large(
    worker_landscape_aggregator, monkeypatch, caplog
):
    error = _message_too_large()

    class FakeHandler:
        def __init__(self, *args, **kwargs):
            pass

        def get_worker_info_task(self):
            raise error

    monkeypatch.setattr(wla_module, "WorkerInfoTasksHandler", FakeHandler)
    with caplog.at_level(logging.ERROR, logger="worker-landscape-tests"):
        with pytest.raises(type(error)):
            worker_landscape_aggregator._get_workers_info(["10.0.0.8:5672"])

    text = caplog.text.lower()
    assert "reachable" in text
    assert "connection error" not in text


def test_wla_datasets_lookup_does_not_hide_message_too_large(
    worker_landscape_aggregator, monkeypatch, caplog
):
    error = _message_too_large()

    class FakeHandler:
        def __init__(self, *args, **kwargs):
            pass

        def get_worker_datasets_per_data_model_task(self):
            raise error

    monkeypatch.setattr(wla_module, "WorkerInfoTasksHandler", FakeHandler)
    with caplog.at_level(logging.ERROR, logger="worker-landscape-tests"):
        with pytest.raises(type(error)):
            worker_landscape_aggregator._get_worker_datasets_per_data_model(
                "10.0.0.8:5672"
            )

    assert "reachable" in caplog.text.lower()


def test_flower_safe_worker_call_logs_message_too_large_as_reachable(caplog):
    error = _message_too_large()
    strategy = FlowerStrategy.__new__(FlowerStrategy)
    strategy._logger = logging.getLogger("flower-message-size")

    def boom():
        raise error

    with caplog.at_level(logging.DEBUG, logger="flower-message-size"):
        asyncio.run(
            strategy._safe_worker_call("garbage collect on global worker", boom)
        )

    text = caplog.text.lower()
    assert "reachable" in text
    assert "garbage collect" in text
    assert "connection error" not in text
    assert "failed to garbage collect" not in text
