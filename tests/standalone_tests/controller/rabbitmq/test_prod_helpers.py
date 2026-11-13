"""Catch production request and recovery-harness regressions without a cluster."""

import asyncio
import json
import subprocess
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import Mock

import pytest

from exaflow import exareme3_algorithm_classes
from exaflow.controller.services.api.analysis_request_dtos import AnalysisRequestDTO
from exaflow.controller.services.api.analysis_request_validator import (
    _validate_required_preprocessing,
)
from tests.algorithm_validation_tests.exareme3 import helpers
from tests.prod_env_tests import test_rabbitmq_analysis as prod


def test_production_pca_payload_satisfies_preprocessing_contract_and_matches_http(
    monkeypatch,
):
    body = prod.payload()
    request = AnalysisRequestDTO.model_validate(body)
    _validate_required_preprocessing(
        exareme3_algorithm_classes["pca"].get_specification(),
        request.preprocessing,
    )
    post = Mock(return_value=SimpleNamespace(status_code=200, content=b"{}"))
    monkeypatch.setattr(helpers.requests, "post", post)
    helpers.analysis_request("pca", body)
    assert json.loads(post.call_args.kwargs["data"]) == body
    steps = [step.name for step in request.preprocessing]
    assert steps.count("missing_values_handler") == 1


@pytest.mark.parametrize("fault", ["connection", "broker_start", "scale_down"])
def test_recovery_waits_for_controller_restoration_even_when_it_fails(
    monkeypatch, fault
):
    commands = []
    starts = 0
    readiness = AsyncMock()
    monkeypatch.setattr(prod, "ready", readiness)

    async def kubectl(*args):
        nonlocal starts
        commands.append(args)
        if fault == "scale_down" and args[0] == "scale" and args[-1] == "--replicas=0":
            raise RuntimeError("scale down failed")
        if args[-1] == "start_app":
            starts += 1
            if fault == "broker_start" and starts == 1:
                raise subprocess.CalledProcessError(1, "kubectl")
        return SimpleNamespace(stdout='{"items": []}')

    monkeypatch.setattr(prod, "kubectl", kubectl)
    sent = SimpleNamespace(correlation_id="recovery-request")
    monkeypatch.setattr(prod, "submit", AsyncMock(return_value=sent))
    queue = SimpleNamespace(get=AsyncMock(return_value=sent))
    delivery_channel = SimpleNamespace(
        get_queue=AsyncMock(return_value=queue), close=AsyncMock()
    )
    connection = SimpleNamespace(
        channel=AsyncMock(return_value=delivery_channel),
        ready=AsyncMock(side_effect=TimeoutError("broker reconnect timed out")),
    )
    broker_exited = []

    @asynccontextmanager
    async def broker():
        try:
            yield connection, object(), object()
        finally:
            broker_exited.append(readiness.await_count)

    monkeypatch.setattr(prod, "broker", broker)

    expected = {
        "connection": TimeoutError,
        "broker_start": subprocess.CalledProcessError,
        "scale_down": RuntimeError,
    }[fault]
    with pytest.raises(expected):
        prod.test_redelivery_survives_broker_restart_and_controller_replacement()

    scale_up = ("scale", "deployment/exaflow-controller-deployment", "--replicas=1")
    rollout = (
        "rollout",
        "status",
        "deployment/exaflow-controller-deployment",
        "--timeout=180s",
    )
    assert commands[-2:] == [scale_up, rollout]
    assert readiness.await_count == 2
    assert broker_exited == [2]
    if fault == "broker_start":
        assert starts == 2
