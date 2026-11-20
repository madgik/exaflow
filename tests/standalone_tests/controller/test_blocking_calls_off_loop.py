import threading

from exaflow.controller.services import analysis_service
from exaflow.controller.services.exareme3 import strategies as exareme3_strategies
from exaflow.controller.services.exareme3.strategies import (
    Exareme3WithAggregationServerStrategy,
)
from exaflow.protos.aggregation_server import aggregation_server_pb2 as agg_pb2
from tests.standalone_tests.controller.analysis_support import ANALYSIS_BODY
from tests.standalone_tests.controller.analysis_support import ResultModel
from tests.standalone_tests.controller.analysis_support import (
    assert_livez_while_blocked,
)
from tests.standalone_tests.controller.analysis_support import build_strategy
from tests.standalone_tests.controller.analysis_support import patch_algorithm


def _patch_entrypoint(monkeypatch, strategy):
    monkeypatch.setattr(
        analysis_service,
        "get_worker_landscape_aggregator",
        lambda: object(),
    )
    monkeypatch.setattr(
        analysis_service, "validate_analysis_request", lambda **kwargs: None
    )
    monkeypatch.setattr(
        analysis_service, "get_algorithm_execution_strategy", lambda dto: strategy
    )


class _FastAlgorithm:
    def __init__(self, **kwargs):
        pass

    def run(self):
        return ResultModel(thread_name="analysis")


def _aggregation_strategy(monkeypatch, configure, cleanup):
    class FakeAggregationClient:
        def __init__(
            self,
            request_id,
            context_id,
            aggregator_dns=None,
            max_message_mb=None,
            tls=None,
        ):
            pass

        def configure(self, num_workers, timeout=None, worker_ids=None):
            return configure(num_workers)

        def cleanup(self, timeout=None):
            return cleanup()

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            pass

    monkeypatch.setattr(
        exareme3_strategies, "ControllerAggregationClient", FakeAggregationClient
    )
    patch_algorithm(monkeypatch, exareme3_strategies, _FastAlgorithm)
    return build_strategy(Exareme3WithAggregationServerStrategy)


def test_livez_during_aggregation_configure(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def configure(num_workers):
        started.set()
        release.wait(timeout=5)
        return agg_pb2.Status.OK

    strategy = _aggregation_strategy(monkeypatch, configure, lambda: agg_pb2.Status.OK)
    _patch_entrypoint(monkeypatch, strategy)
    try:
        assert_livez_while_blocked(
            started,
            release,
            lambda client: client.post("/analysis", json=ANALYSIS_BODY),
        )
    finally:
        release.set()


def test_livez_during_aggregation_cleanup(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def cleanup():
        started.set()
        release.wait(timeout=5)
        return agg_pb2.Status.OK

    strategy = _aggregation_strategy(
        monkeypatch, lambda num_workers: agg_pb2.Status.OK, cleanup
    )
    _patch_entrypoint(monkeypatch, strategy)
    try:
        assert_livez_while_blocked(
            started,
            release,
            lambda client: client.post("/analysis", json=ANALYSIS_BODY),
        )
    finally:
        release.set()
