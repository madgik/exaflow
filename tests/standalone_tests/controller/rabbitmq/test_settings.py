import asyncio
import re
import shutil
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from exaflow import AttrDict
from exaflow.controller.http import app as app_module
from exaflow.controller.services import controller_shutdown as budget
from exaflow.controller.services.rabbitmq.settings import RabbitMQSettings


def test_disabled_broker_settings_do_not_block_controller_startup(monkeypatch):
    monkeypatch.setattr(
        app_module,
        "ctrl_config",
        AttrDict(
            {
                "rabbitmq": {
                    "enabled": "false",
                    "url": "invalid",
                    "operation_timeout": 10,
                    "request_queue": "",
                    "publish_timeout": -1,
                }
            }
        ),
    )
    startup, shutdown = AsyncMock(), AsyncMock()
    monkeypatch.setattr(app_module.endpoints, "startup", startup)
    monkeypatch.setattr(app_module.endpoints, "shutdown", shutdown)

    async def scenario():
        async with app_module.lifespan(app_module.app):
            assert not app_module.app.state.rabbitmq.settings.enabled

    previous = getattr(app_module.app.state, "rabbitmq", None)
    try:
        asyncio.run(scenario())
    finally:
        app_module.app.state.rabbitmq = previous
    startup.assert_awaited_once()
    shutdown.assert_awaited_once()


@pytest.mark.parametrize(
    "overrides",
    [
        {"url": "invalid"},
        {"operation_timeout": 10},
        {"publish_timeout": 0},
        {"shutdown_drain_timeout": 30},
    ],
)
def test_enabled_configuration_is_validated(overrides):
    with pytest.raises(ValidationError):
        RabbitMQSettings.from_config(
            {"enabled": True, "url": "amqp://localhost", **overrides}
        )


def test_publish_timeout_is_independent_and_shutdown_has_headroom():
    settings = RabbitMQSettings(publish_timeout=60)
    assert settings.operation_timeout == 5
    assert settings.publish_timeout == 60
    worst_case = (
        3 * settings.operation_timeout
        + settings.shutdown_drain_timeout
        + budget.SHUTDOWN_CLEANUP_TIMEOUT
        + budget.SHUTDOWN_RPC_CANCEL_TIMEOUT
    )
    assert worst_case <= budget.SHUTDOWN_TIMEOUT - 10


@pytest.mark.skipif(shutil.which("helm") is None, reason="helm is not installed")
def test_helm_rabbitmq_values_keep_decimal_integers_and_fractional_timeouts():
    root = Path(__file__).resolve().parents[4]
    rendered = subprocess.check_output(
        [
            "helm",
            "template",
            str(root / "kubernetes"),
            "--show-only",
            "templates/exaflow-controller.yaml",
            "--set",
            "rabbitmq.enabled=true",
            "--set",
            "rabbitmq.operation_timeout=2.5",
            "--set",
            "rabbitmq.publish_timeout=45.5",
        ],
        text=True,
    )
    expected = {
        "MAX_RESULT_BYTES": "16777216",
        "QUEUE_MAX_BYTES": "268435456",
        "REPLY_TTL_MS": "86400000",
        "DEAD_LETTER_TTL_MS": "604800000",
        "OPERATION_TIMEOUT": "2.5",
        "PUBLISH_TIMEOUT": "45.5",
        "SHUTDOWN_DRAIN_TIMEOUT": "20",
    }
    for key, value in expected.items():
        assert re.search(rf'name: RABBITMQ_{key}\s+value: "{value}"', rendered)
