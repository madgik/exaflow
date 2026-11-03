"""Analysis orchestration lives in a service function, not the route."""

import asyncio
from contextlib import asynccontextmanager

import pytest
from exaflow.controller.services.analysis_service import run_analysis

from exaflow.controller.services import analysis_service
from exaflow.controller.services.api.analysis_request_validator import BadRequest

ANALYSIS_REQUEST = {
    "inputdata": {
        "data_model": "data_model",
        "datasets": ["dataset"],
        "variables": ["variable"],
    },
    "algorithm": {"name": "algorithm", "y": ["variable"]},
}


def test_malformed_analysis_body_is_a_bad_request():
    with pytest.raises(BadRequest) as caught:
        asyncio.run(run_analysis({"foo": "bar"}))

    message = str(caught.value)
    assert message.startswith("Analysis execution request malformed:")
    assert "request received:{'foo': 'bar'}" in message


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
    monkeypatch.setattr(analysis_service, "analysis_slot", slot)

    body = asyncio.run(
        run_analysis(
            ANALYSIS_REQUEST,
            validate_analysis_request=validate,
            get_algorithm_execution_strategy=lambda dto: _Strategy(),
            get_worker_landscape_aggregator=lambda: aggregator,
        )
    )

    assert body == '{"result": 1}'
    assert entered["value"]
    assert seen["dto"].algorithm.name == "algorithm"
    assert seen["aggregator"] is aggregator
