"""Cleanup must not delete a context this caller does not own.

A duplicate Configure that times out or loses its response never observes
ALREADY_CONFIGURED. Treating that failure as ownership makes the duplicate's
finally block Cleanup the request id, and the server then aborts the analysis
that actually created the context.
"""

import asyncio
import logging

import grpc
import pytest

from exaflow.aggregation_clients.controller_aggregation_client import (
    ControllerAggregationClient,
)
from exaflow.aggregation_server.server import AggregationServer
from exaflow.controller.services.errors import ControllerShuttingDownError
from exaflow.controller.services.exareme3 import strategies
from exaflow.protos.aggregation_server import aggregation_server_pb2 as pb2
from tests.standalone_tests.controller.analysis_support import build_strategy


class _MetadataContext:
    def __init__(self, metadata=()):
        self._metadata = tuple(metadata or ())

    def invocation_metadata(self):
        return self._metadata

    def abort(self, code, details):
        raise AssertionError(details)


class _Done:
    def __init__(self, response):
        self._response = response

    def result(self):
        return self._response

    def cancel(self):
        return False

    def add_done_callback(self, callback):
        callback(self)


class _LostResponse:
    """The RPC completed on the server and the response never arrived."""

    def result(self):
        raise grpc.FutureCancelledError()

    def cancel(self):
        return False

    def add_done_callback(self, callback):
        callback(self)


def _stub(configure_future, cleanup):
    return type(
        "Stub",
        (),
        {
            "Configure": type(
                "Configure", (), {"future": staticmethod(configure_future)}
            ),
            "Cleanup": staticmethod(cleanup),
        },
    )()


def _wire(client, configure_future, cleanup):
    client._stub = _stub(configure_future, cleanup)
    return client


def test_duplicate_lost_configure_does_not_remove_the_owners_context(monkeypatch):
    """A lost duplicate Configure must leave the original context in place."""
    servicer = AggregationServer()
    request_id = "shared-request"

    def owner_future(request, timeout=None, metadata=None):
        response = servicer.Configure(request, _MetadataContext(metadata))
        return _Done(response)

    def cleanup(request, timeout=None, metadata=None):
        return servicer.Cleanup(request, _MetadataContext(metadata))

    owner = _wire(
        ControllerAggregationClient(request_id),
        owner_future,
        cleanup,
    )
    assert owner.configure(2, timeout=5) == pb2.Status.OK
    assert request_id in servicer.aggregation_contexts

    def lost_future(request, timeout=None, metadata=None):
        # The server still answers ALREADY_CONFIGURED; the caller never sees it.
        servicer.Configure(request, _MetadataContext(metadata))
        return _LostResponse()

    def factory(request_id, aggregator_dns=None, max_message_mb=None):
        return _wire(
            ControllerAggregationClient(
                request_id,
                aggregator_dns=aggregator_dns,
                max_message_mb=max_message_mb,
            ),
            lost_future,
            cleanup,
        )

    monkeypatch.setattr(strategies, "ControllerAggregationClient", factory)
    strategy = build_strategy(strategies.Exareme3WithAggregationServerStrategy)
    strategy._request_id = request_id
    strategy._logger = logging.getLogger("aggregation-ownership")

    with pytest.raises(ControllerShuttingDownError):
        asyncio.run(strategy.execute())

    assert request_id in servicer.aggregation_contexts
    assert servicer.aggregation_contexts[request_id].error is None
    assert owner.cleanup(timeout=5) == pb2.Status.OK
    assert request_id not in servicer.aggregation_contexts


def test_owner_cleans_up_when_its_configure_response_is_lost():
    """The caller that created the context still removes it if the OK is lost."""
    servicer = AggregationServer()
    request_id = "owner-lost-ok"

    def lost_future(request, timeout=None, metadata=None):
        servicer.Configure(request, _MetadataContext(metadata))
        return _LostResponse()

    def cleanup(request, timeout=None, metadata=None):
        return servicer.Cleanup(request, _MetadataContext(metadata))

    client = _wire(ControllerAggregationClient(request_id), lost_future, cleanup)
    with pytest.raises(ControllerShuttingDownError):
        client.configure(2, timeout=5)
    assert request_id in servicer.aggregation_contexts

    assert client.cleanup(timeout=5) == pb2.Status.OK
    assert request_id not in servicer.aggregation_contexts
