import asyncio
import logging

import pytest

from exaflow.controller.services.exareme3 import strategies
from exaflow.protos.aggregation_server import aggregation_server_pb2 as agg_pb2


class _RecordingAggregationClient:
    def __init__(self, request_id, aggregator_dns=None, max_message_mb=None):
        self.calls = []

    def configure(self, num_workers):
        self.calls.append("configure")
        return agg_pb2.Status.OK

    def cleanup(self):
        self.calls.append("cleanup")
        return agg_pb2.Status.OK

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.calls.append("close")


def test_aggregation_strategy_cleans_up_and_closes_client_when_algorithm_fails(
    monkeypatch,
):
    created = []

    def factory(request_id, aggregator_dns=None, max_message_mb=None):
        client = _RecordingAggregationClient(
            request_id,
            aggregator_dns=aggregator_dns,
            max_message_mb=max_message_mb,
        )
        created.append(client)
        return client

    async def failing_execute(self):
        raise ValueError("worker failed")

    monkeypatch.setattr(strategies, "ControllerAggregationClient", factory)
    monkeypatch.setattr(strategies.Exareme3Strategy, "execute", failing_execute)
    strategy = object.__new__(strategies.Exareme3WithAggregationServerStrategy)
    strategy._request_id = "req"
    strategy._local_worker_tasks_handlers = [object(), object()]
    strategy._logger = logging.getLogger(__name__)

    with pytest.raises(ValueError, match="worker failed"):
        asyncio.run(strategy.execute())

    assert created[-1].calls == [
        "configure",
        "cleanup",
        "close",
    ]
