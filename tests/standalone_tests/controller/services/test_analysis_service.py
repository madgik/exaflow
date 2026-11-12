"""Analysis orchestration lives in a service function, not the route."""

import asyncio
from contextlib import asynccontextmanager

from exaflow.controller.services import analysis_service
from exaflow.controller.services.analysis_service import run_analysis
from tests.standalone_tests.controller.analysis_support import analysis_request


def test_run_analysis_validates_then_executes_inside_a_slot(monkeypatch):
    entered = {"value": False}
    seen = {}

    @asynccontextmanager
    async def slot():
        entered["value"] = True
        yield

    class _Strategy:
        async def execute(self):
            assert entered["value"]
            return '{"result": 1}'

    def validate(**kwargs):
        seen["dto"] = kwargs["analysis_request_dto"]
        seen["aggregator"] = kwargs["worker_landscape_aggregator"]

    aggregator = object()
    request = analysis_request()
    monkeypatch.setattr(analysis_service, "analysis_slot", slot)
    monkeypatch.setattr(analysis_service, "validate_analysis_request", validate)
    monkeypatch.setattr(
        analysis_service, "get_algorithm_execution_strategy", lambda dto: _Strategy()
    )
    monkeypatch.setattr(
        analysis_service, "get_worker_landscape_aggregator", lambda: aggregator
    )

    body = asyncio.run(run_analysis(request))

    assert body == '{"result": 1}'
    assert entered["value"]
    assert seen["dto"] is request
    assert seen["aggregator"] is aggregator
