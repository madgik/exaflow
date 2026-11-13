"""Real AMQP transport against the kind deployment, never a mocked broker."""

import asyncio
import json
import os
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import aio_pika
import numpy as np
import pytest
import requests

from tests.algorithm_validation_tests.exareme3.helpers import analysis_request
from tests.algorithm_validation_tests.exareme3.helpers import (
    build_analysis_request_payload,
)
from tests.algorithm_validation_tests.exareme3.helpers import parse_response

pytestmark = pytest.mark.skipif(
    not os.environ.get("RABBITMQ_TEST_URL"),
    reason="Requires the production test RabbitMQ deployment and RABBITMQ_TEST_URL",
)


def payload():
    cases = json.loads(
        (Path(__file__).parent / "expected/pca_expected.json").read_text()
    )
    body = build_analysis_request_payload("pca", cases["test_cases"][0]["input"])
    body["request_id"] = uuid4().hex
    return body


async def ready():
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            response = await asyncio.to_thread(
                requests.get,
                "http://127.0.0.1:5100/healthcheck/rabbitmq",
                timeout=5,
            )
            if response.status_code == 200 and response.json()["consuming"]:
                return
        except requests.RequestException:
            pass
        await asyncio.sleep(0.5)
    raise AssertionError("Controller AMQP consumer did not become ready")


@asynccontextmanager
async def broker():
    connection = await aio_pika.connect_robust(
        os.environ["RABBITMQ_TEST_URL"], timeout=10
    )
    channel = await connection.channel(publisher_confirms=True, on_return_raises=True)
    replies = await channel.declare_queue(
        f"exaflow.tests.replies.{uuid4().hex}",
        durable=True,
        arguments={
            "x-message-ttl": 86400000,
            "x-expires": 3600000,
            "x-max-length-bytes": 268435456,
            "x-overflow": "reject-publish",
        },
    )
    try:
        yield connection, channel, replies
    finally:
        try:
            await asyncio.wait_for(replies.delete(), 10)
        finally:
            await asyncio.wait_for(connection.close(), 10)


async def submit(channel, replies, body, **properties):
    request_id = body.get("request_id", uuid4().hex)
    message = aio_pika.Message(
        json.dumps(body).encode(),
        content_type="application/json",
        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        correlation_id=request_id,
        reply_to=replies.name,
        message_id=uuid4().hex,
    )
    for key, value in properties.items():
        setattr(message, key, value)
    await channel.default_exchange.publish(
        message,
        routing_key="exaflow.analysis.requests",
        mandatory=True,
        timeout=10,
    )
    return message


async def receive(queue, predicate=lambda message: True, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        message = await queue.get(fail=False, timeout=5)
        if message is not None:
            if predicate(message):
                await message.ack()
                return message
            # Leave unrelated dead letters unacknowledged; channel closure restores them.
        await asyncio.sleep(0.1)
    raise AssertionError("Expected broker message did not arrive")


def test_queue_pca_matches_http_and_repeat_request_gets_fresh_aggregation_context():
    async def scenario():
        await ready()
        body = payload()
        case = {k: v for k, v in body.items() if k != "request_id"}
        expected = parse_response(
            await asyncio.to_thread(analysis_request, "pca", case)
        )
        async with broker() as (_, channel, replies):
            for _ in range(2):
                await submit(channel, replies, body)
                message = await receive(replies)
                assert message.correlation_id == body["request_id"]
                result = json.loads(message.body)
                assert result["status"] == "succeeded", result
                actual = result["result"]
                assert actual["n_obs"] == expected["n_obs"]
                assert actual["variables"] == expected["variables"]
                np.testing.assert_allclose(
                    [c["explained_variance"] for c in actual["components"]],
                    [c["explained_variance"] for c in expected["components"]],
                    rtol=1e-6,
                    atol=1e-9,
                )
        assert (
            await asyncio.to_thread(
                requests.get, "http://127.0.0.1:5100/livez", timeout=5
            )
        ).status_code == 200

    asyncio.run(scenario())


@pytest.mark.parametrize("malformed", [False, True])
def test_queue_validation_failures_are_terminal_replies(malformed):
    async def scenario():
        await ready()
        body = payload()
        if malformed:
            body = {"request_id": body["request_id"]}
        else:
            body["algorithm"]["name"] = "does_not_exist"
        async with broker() as (_, channel, replies):
            await submit(channel, replies, body)
            reply = json.loads((await receive(replies)).body)
            assert reply["status"] == "failed"
            assert reply["error"]["code"] == (
                "invalid_request" if malformed else "BadRequest"
            )

    asyncio.run(scenario())


@pytest.mark.parametrize("missing_reply_destination", [False, True])
def test_bad_routing_goes_to_real_dead_letter_queue(missing_reply_destination):
    async def scenario():
        await ready()
        async with broker() as (_, channel, replies):
            properties = (
                {"reply_to": f"absent.{uuid4().hex}"}
                if missing_reply_destination
                else {"correlation_id": None}
            )
            # Structurally invalid body produces a quick failure for an unroutable reply.
            sent = await submit(channel, replies, {}, **properties)
            dead = await channel.get_queue("exaflow.analysis.dead", ensure=True)
            received = await receive(
                dead, lambda m: m.message_id == sent.message_id, timeout=30
            )
            assert received.body == sent.body
            assert received.headers["x-death"]

    asyncio.run(scenario())


async def kubectl(*args):
    return await asyncio.to_thread(
        subprocess.run,
        ["kubectl", *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )


async def restore_controller():
    await kubectl("scale", "deployment/exaflow-controller-deployment", "--replicas=1")
    await kubectl(
        "rollout",
        "status",
        "deployment/exaflow-controller-deployment",
        "--timeout=180s",
    )
    await ready()


@pytest.mark.skipif(
    os.environ.get("RABBITMQ_TEST_RECOVERY") != "1",
    reason="Requires exclusive ownership of the production test deployment",
)
def test_redelivery_survives_broker_restart_and_controller_replacement():
    async def scenario():
        await ready()
        async with broker() as (connection, channel, replies):
            broker_stopped = False
            try:
                await kubectl(
                    "scale", "deployment/exaflow-controller-deployment", "--replicas=0"
                )
                deadline = time.monotonic() + 180
                while True:
                    pods = await kubectl(
                        "get", "pods", "-l", "app=exaflow-controller", "-o", "json"
                    )
                    if not json.loads(pods.stdout)["items"]:
                        break
                    assert time.monotonic() < deadline, "Controller did not stop"
                    await asyncio.sleep(0.5)
                sent = await submit(channel, replies, payload())
                delivery_channel = await connection.channel()
                requests_queue = await delivery_channel.get_queue(
                    "exaflow.analysis.requests", ensure=True
                )
                # Simulate loss of an unacknowledged delivery before restarting the broker.
                held = await requests_queue.get(timeout=5)
                assert held.correlation_id == sent.correlation_id
                await delivery_channel.close()
                broker_stopped = True
                await kubectl(
                    "exec", "exaflow-rabbitmq-0", "--", "rabbitmqctl", "stop_app"
                )
                await kubectl(
                    "exec", "exaflow-rabbitmq-0", "--", "rabbitmqctl", "start_app"
                )
                broker_stopped = False
                await kubectl(
                    "rollout",
                    "status",
                    "statefulset/exaflow-rabbitmq",
                    "--timeout=120s",
                )
                await asyncio.wait_for(connection.ready(), 30)
            finally:
                try:
                    if broker_stopped:
                        await kubectl(
                            "exec",
                            "exaflow-rabbitmq-0",
                            "--",
                            "rabbitmqctl",
                            "start_app",
                        )
                finally:
                    await restore_controller()
            result = await receive(replies)
            assert result.correlation_id == sent.correlation_id
            assert json.loads(result.body)["status"] == "succeeded"

    asyncio.run(scenario())
