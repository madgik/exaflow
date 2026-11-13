import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aio_pika.exceptions import PublishError
from pamqp.commands import Basic

from exaflow.controller.services.rabbitmq import consumer as module
from exaflow.controller.services.rabbitmq.consumer import AnalysisConsumer
from exaflow.controller.services.rabbitmq.settings import RabbitMQSettings
from tests.standalone_tests.controller.analysis_support import ANALYSIS_BODY


def resources(body=None, **metadata):
    events = []
    message = SimpleNamespace(
        body=json.dumps(ANALYSIS_BODY if body is None else body).encode(),
        correlation_id="request-1",
        reply_to="replies",
        ack=AsyncMock(side_effect=lambda: events.append("ack")),
        reject=AsyncMock(side_effect=lambda **kw: events.append("reject")),
    )
    for key, value in metadata.items():
        setattr(message, key, value)
    channel = SimpleNamespace(
        default_exchange=SimpleNamespace(
            publish=AsyncMock(side_effect=lambda *a, **kw: events.append("publish"))
        ),
        close=AsyncMock(),
    )
    return message, channel, events


def consumer(**settings):
    return AnalysisConsumer(RabbitMQSettings(**settings), lambda: True)


def test_success_confirms_reply_before_acknowledging(monkeypatch):
    execute = AsyncMock(return_value='{"value": 7}')
    monkeypatch.setattr(module.analysis_service, "run_analysis", execute)
    message, channel, events = resources()
    runtime = consumer()
    asyncio.run(runtime.handle(message, channel))
    assert events == ["publish", "ack"]
    request = execute.call_args.args[0]
    assert request.request_id == "request-1"
    assert execute.call_args.kwargs == {"wait_for_capacity": True}
    outgoing = channel.default_exchange.publish.call_args.args[0]
    assert json.loads(outgoing.body) == {"status": "succeeded", "result": {"value": 7}}
    assert outgoing.correlation_id == "request-1"
    assert outgoing.delivery_mode == 2
    assert not runtime.executions


@pytest.mark.parametrize("body", [{}, {**ANALYSIS_BODY, "request_id": "conflict"}])
def test_invalid_body_replies_without_execution(monkeypatch, body):
    execute = AsyncMock()
    monkeypatch.setattr(module.analysis_service, "run_analysis", execute)
    message, channel, events = resources(body)
    asyncio.run(consumer().handle(message, channel))
    execute.assert_not_called()
    reply = json.loads(channel.default_exchange.publish.call_args.args[0].body)
    assert reply["error"]["code"] == "invalid_request"
    assert events == ["publish", "ack"]


@pytest.mark.parametrize("field", ["correlation_id", "reply_to"])
def test_missing_routing_metadata_dead_letters(field):
    message, channel, events = resources(**{field: None})
    asyncio.run(consumer().handle(message, channel))
    assert events == ["reject"]
    message.reject.assert_awaited_once_with(requeue=False)
    channel.default_exchange.publish.assert_not_called()


def test_transient_publish_failure_leaves_request_unacked(monkeypatch):
    monkeypatch.setattr(
        module.analysis_service, "run_analysis", AsyncMock(return_value="{}")
    )
    message, channel, _ = resources()
    channel.default_exchange.publish.side_effect = TimeoutError()
    asyncio.run(consumer().handle(message, channel))
    message.ack.assert_not_called()
    message.reject.assert_not_called()
    channel.close.assert_awaited_once()


def test_unroutable_reply_dead_letters_request(monkeypatch):
    monkeypatch.setattr(
        module.analysis_service, "run_analysis", AsyncMock(return_value="{}")
    )
    message, channel, _ = resources()
    channel.default_exchange.publish.side_effect = PublishError(
        SimpleNamespace(
            delivery=Basic.Return(
                reply_code=312, reply_text="NO_ROUTE", routing_key="replies"
            )
        ),
        Basic.Return(),
    )
    asyncio.run(consumer().handle(message, channel))
    message.reject.assert_awaited_once_with(requeue=False)
    message.ack.assert_not_called()


def test_active_redelivery_shares_execution(monkeypatch):
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        async def execute(*args, **kw):
            entered.set()
            await release.wait()
            return "{}"

        mock = AsyncMock(side_effect=execute)
        monkeypatch.setattr(module.analysis_service, "run_analysis", mock)
        runtime = consumer()
        first, ch1, _ = resources()
        second, ch2, _ = resources()
        task1 = asyncio.create_task(runtime.handle(first, ch1))
        await entered.wait()
        task2 = asyncio.create_task(runtime.handle(second, ch2))
        await asyncio.sleep(0)
        release.set()
        await asyncio.gather(task1, task2)
        assert mock.await_count == 1
        first.ack.assert_awaited_once()
        second.ack.assert_awaited_once()

    asyncio.run(scenario())


def test_request_and_result_size_limits(monkeypatch):
    message, channel, _ = resources()
    monkeypatch.setattr(module, "MAX_REQUEST_BYTES", 1)
    asyncio.run(consumer().handle(message, channel))
    assert (
        json.loads(channel.default_exchange.publish.call_args.args[0].body)["error"][
            "code"
        ]
        == "request_too_large"
    )
    monkeypatch.setattr(module, "MAX_REQUEST_BYTES", 16 * 1024 * 1024)
    monkeypatch.setattr(
        module.analysis_service,
        "run_analysis",
        AsyncMock(return_value=json.dumps({"x": "x" * 2000})),
    )
    message, channel, _ = resources()
    asyncio.run(consumer(max_result_bytes=1024).handle(message, channel))
    assert (
        json.loads(channel.default_exchange.publish.call_args.args[0].body)["error"][
            "code"
        ]
        == "result_too_large"
    )


def test_unexpected_errors_do_not_expose_exception_details(monkeypatch):
    monkeypatch.setattr(
        module.analysis_service,
        "run_analysis",
        AsyncMock(side_effect=RuntimeError("secret payload")),
    )
    message, channel, _ = resources()
    asyncio.run(consumer().handle(message, channel))
    body = channel.default_exchange.publish.call_args.args[0].body
    assert b"internal_error" in body and b"secret payload" not in body


def test_shutdown_error_is_redelivered_not_terminal(monkeypatch):
    monkeypatch.setattr(
        module.analysis_service,
        "run_analysis",
        AsyncMock(side_effect=module.ControllerShuttingDownError()),
    )
    message, channel, _ = resources()
    asyncio.run(consumer().handle(message, channel))
    message.ack.assert_not_called()
    channel.default_exchange.publish.assert_not_called()


def test_drain_stops_deliveries_before_waiting_for_work():
    async def scenario():
        runtime = consumer(shutdown_drain_timeout=0)
        runtime.iterator = SimpleNamespace(close=AsyncMock())
        job = asyncio.create_task(asyncio.Event().wait())
        runtime.deliveries.add(job)
        await runtime.drain()
        runtime.iterator.close.assert_awaited_once()
        assert not job.cancelled()
        await runtime.close()
        assert job.cancelled()

    asyncio.run(scenario())


def test_broker_outage_retries_and_stays_independent_of_http(monkeypatch):
    async def scenario():
        disconnected = asyncio.Event()

        async def connect(*args, **kwargs):
            disconnected.set()
            raise ConnectionError("amqp://private-credentials")

        monkeypatch.setattr(module.aio_pika, "connect", connect)
        runtime = consumer(enabled=True, url="amqp://localhost")
        runtime.start()
        await disconnected.wait()
        await asyncio.sleep(0)
        assert runtime.health()["last_error"] == "ConnectionError"
        assert not runtime.health()["consuming"]
        await runtime.drain()
        await runtime.close()

    asyncio.run(scenario())


def test_configuration_validation_redacts_credentials():
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as exc:
        RabbitMQSettings(enabled=True, url="bad://secret-password")
    assert "secret-password" not in str(exc.value)


def test_supervisor_bounds_delivery_handlers_and_drains_without_closing_active_replies(
    monkeypatch,
):
    async def scenario():
        monkeypatch.setattr(module, "max_concurrent_experiments", lambda: 2)
        messages = [resources()[0] for _ in range(3)]
        for index, message in enumerate(messages):
            message.correlation_id = f"job-{index}"
        released = asyncio.Event()
        started = asyncio.Event()
        completed = asyncio.Event()
        calls = 0

        async def execute(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                started.set()
            await released.wait()
            if calls == 3:
                completed.set()
            return "{}"

        monkeypatch.setattr(
            module.analysis_service, "run_analysis", AsyncMock(side_effect=execute)
        )

        class Iterator:
            def __init__(self):
                self.items = iter(messages)
                self.closed = asyncio.Event()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                await self.close()

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.closed.is_set():
                    raise StopAsyncIteration
                item = next(self.items, None)
                if item is not None:
                    return item
                await self.closed.wait()
                raise StopAsyncIteration

            async def close(self):
                self.closed.set()

        iterator = Iterator()
        queue = SimpleNamespace(
            iterator=lambda: iterator,
            declare=AsyncMock(return_value=SimpleNamespace(consumer_count=1)),
        )
        channel = SimpleNamespace(
            set_qos=AsyncMock(),
            declare_queue=AsyncMock(return_value=queue),
            default_exchange=SimpleNamespace(publish=AsyncMock()),
            close=AsyncMock(),
        )

        class Connection:
            is_closed = False

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                await self.close()

            async def channel(self, **kwargs):
                return channel

            async def close(self):
                self.is_closed = True

        connection = Connection()
        monkeypatch.setattr(
            module.aio_pika, "connect", AsyncMock(return_value=connection)
        )
        runtime = consumer(enabled=True, url="amqp://localhost")
        runtime.start()
        await asyncio.wait_for(started.wait(), 2)
        assert runtime.health()["active_jobs"] == 2
        assert len(runtime.deliveries) == 2
        released.set()
        await asyncio.wait_for(completed.wait(), 2)
        await runtime.drain()
        assert not connection.is_closed
        for message in messages:
            message.ack.assert_awaited_once()
        await runtime.close()
        assert connection.is_closed

    asyncio.run(scenario())


def test_cancelled_subscription_interrupts_waiting_iterator(monkeypatch):
    async def scenario():
        runtime = consumer(enabled=True, url="amqp://localhost")
        runtime.consuming = True
        queue = SimpleNamespace(
            declare=AsyncMock(return_value=SimpleNamespace(consumer_count=0))
        )
        interrupted = asyncio.Event()

        async def wait_for_message():
            try:
                await asyncio.Event().wait()
            finally:
                interrupted.set()

        with pytest.raises(ExceptionGroup) as exc:
            async with asyncio.TaskGroup() as tasks:
                tasks.create_task(wait_for_message())
                tasks.create_task(runtime._monitor_subscription(queue))
        assert isinstance(exc.value.exceptions[0], ConnectionError)
        assert interrupted.is_set()
        assert not runtime.health()["consuming"]

    asyncio.run(scenario())


def test_supervisor_reconnects_after_subscription_disappears(monkeypatch):
    async def scenario():
        restored = asyncio.Event()
        connections = []

        class Iterator:
            def __init__(self, second):
                self.closed = asyncio.Event()
                self.second = second

            async def __aenter__(self):
                if self.second:
                    restored.set()
                return self

            async def __aexit__(self, *args):
                await self.close()

            def __aiter__(self):
                return self

            async def __anext__(self):
                await self.closed.wait()
                raise StopAsyncIteration

            async def close(self):
                self.closed.set()

        class Connection:
            is_closed = False

            def __init__(self, second):
                iterator = Iterator(second)
                queue = SimpleNamespace(
                    iterator=lambda: iterator,
                    declare=AsyncMock(
                        return_value=SimpleNamespace(consumer_count=int(second))
                    ),
                )
                self.amqp_channel = SimpleNamespace(
                    set_qos=AsyncMock(), declare_queue=AsyncMock(return_value=queue)
                )

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                self.is_closed = True

            async def channel(self, **kwargs):
                return self.amqp_channel

            async def close(self):
                self.is_closed = True

        async def connect(*args, **kwargs):
            connection = Connection(bool(connections))
            connections.append(connection)
            return connection

        monkeypatch.setattr(module.aio_pika, "connect", connect)
        runtime = consumer(enabled=True, url="amqp://localhost")
        runtime.start()
        try:
            await asyncio.wait_for(restored.wait(), 4)
            assert len(connections) == 2
            assert connections[0].is_closed
            assert runtime.health()["consuming"]
        finally:
            await runtime.drain()
            await runtime.close()

    asyncio.run(scenario())
