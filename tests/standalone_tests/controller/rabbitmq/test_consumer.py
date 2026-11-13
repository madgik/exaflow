import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import Mock

import pytest
from aio_pika.exceptions import ChannelNotFoundEntity
from aio_pika.exceptions import ChannelPreconditionFailed
from aio_pika.exceptions import DeliveryError
from aio_pika.exceptions import PublishError
from pamqp.commands import Basic

from exaflow.controller.services.rabbitmq import consumer as module
from exaflow.controller.services.rabbitmq.consumer import AnalysisConsumer
from exaflow.controller.services.rabbitmq.settings import RabbitMQSettings
from tests.standalone_tests.controller.analysis_support import ANALYSIS_BODY
from tests.standalone_tests.controller.analysis_support import INVALID_REQUEST_IDS
from tests.standalone_tests.controller.analysis_support import VALID_REQUEST_IDS


def resources(body=None, **metadata):
    events = []
    message = SimpleNamespace(
        body=json.dumps(ANALYSIS_BODY if body is None else body).encode(),
        correlation_id="request-1",
        reply_to="replies",
        redelivered=False,
        channel=SimpleNamespace(is_closed=False, close=AsyncMock()),
        ack=AsyncMock(side_effect=lambda: events.append("ack")),
        reject=AsyncMock(side_effect=lambda **kw: events.append("reject")),
    )
    for key, value in metadata.items():
        setattr(message, key, value)
    reply_channel = SimpleNamespace(
        declare_queue=AsyncMock(),
        default_exchange=SimpleNamespace(
            publish=AsyncMock(side_effect=lambda *a, **kw: events.append("publish"))
        ),
        close=AsyncMock(),
    )
    channel = SimpleNamespace(
        is_closed=False,
        channel=AsyncMock(return_value=reply_channel),
        default_exchange=reply_channel.default_exchange,
        reply_channel=reply_channel,
        close=AsyncMock(),
    )
    return message, channel, events


def consumer(**settings):
    return AnalysisConsumer(RabbitMQSettings(**settings), lambda: True)


def test_success_confirms_reply_before_acknowledging(monkeypatch):
    execute = AsyncMock(return_value='{"value": 7}')
    monkeypatch.setattr(module.analysis_service, "run_analysis", execute)
    message, channel, events = resources()
    runtime = consumer(publish_timeout=45)
    asyncio.run(runtime.handle(message, channel))
    assert events == ["publish", "ack"]
    request = execute.call_args.args[0]
    assert request.request_id == "request-1"
    assert execute.call_args.kwargs == {"wait_for_capacity": True}
    outgoing = channel.default_exchange.publish.call_args.args[0]
    assert json.loads(outgoing.body) == {"status": "succeeded", "result": {"value": 7}}
    assert outgoing.correlation_id == "request-1"
    assert outgoing.delivery_mode == 2
    assert channel.default_exchange.publish.call_args.kwargs["timeout"] == 45
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


@pytest.mark.parametrize("request_id", INVALID_REQUEST_IDS[1:])
@pytest.mark.parametrize("body_id", ["omitted", "null", "same"])
def test_invalid_correlation_id_cannot_bypass_dto_validation(
    monkeypatch, request_id, body_id
):
    execute = AsyncMock()
    monkeypatch.setattr(module.analysis_service, "run_analysis", execute)
    body = dict(ANALYSIS_BODY)
    if body_id != "omitted":
        body["request_id"] = None if body_id == "null" else request_id
    message, connection, events = resources(body, correlation_id=request_id)
    runtime = consumer()
    asyncio.run(runtime.handle(message, connection))
    reply = connection.default_exchange.publish.call_args.args[0]
    assert json.loads(reply.body)["error"]["code"] == "invalid_request"
    assert reply.correlation_id == request_id
    execute.assert_not_called()
    assert not runtime.executions
    assert not runtime.completed.entries
    assert events == ["publish", "ack"]


@pytest.mark.parametrize("request_id", INVALID_REQUEST_IDS)
def test_invalid_body_request_id_is_rejected_with_valid_correlation_id(
    monkeypatch, request_id
):
    execute = AsyncMock()
    monkeypatch.setattr(module.analysis_service, "run_analysis", execute)
    message, connection, _ = resources({**ANALYSIS_BODY, "request_id": request_id})
    runtime = consumer()
    asyncio.run(runtime.handle(message, connection))
    reply = connection.default_exchange.publish.call_args.args[0]
    assert json.loads(reply.body)["error"]["code"] == "invalid_request"
    execute.assert_not_called()
    assert not runtime.executions
    assert not runtime.completed.entries


@pytest.mark.parametrize("request_id", VALID_REQUEST_IDS)
@pytest.mark.parametrize("body_id", ["omitted", "null", "same"])
def test_valid_correlation_id_reaches_execution_and_reply_unchanged(
    monkeypatch, request_id, body_id
):
    execute = AsyncMock(return_value="{}")
    monkeypatch.setattr(module.analysis_service, "run_analysis", execute)
    body = dict(ANALYSIS_BODY)
    if body_id != "omitted":
        body["request_id"] = None if body_id == "null" else request_id
    message, connection, _ = resources(body, correlation_id=request_id)
    asyncio.run(consumer().handle(message, connection))
    assert execute.await_args.args[0].request_id == request_id
    assert (
        connection.default_exchange.publish.call_args.args[0].correlation_id
        == request_id
    )


def test_invalid_correlation_id_is_rejected_before_retained_reply_lookup(monkeypatch):
    execute = AsyncMock()
    monkeypatch.setattr(module.analysis_service, "run_analysis", execute)
    message, connection, _ = resources(correlation_id="request\n", redelivered=True)
    runtime = consumer()
    runtime.completed.get = Mock(
        side_effect=AssertionError("Invalid ID reached reply cache")
    )
    asyncio.run(runtime.handle(message, connection))
    runtime.completed.get.assert_not_called()
    execute.assert_not_called()
    reply = connection.default_exchange.publish.call_args.args[0]
    assert json.loads(reply.body)["error"]["code"] == "invalid_request"


@pytest.mark.parametrize("field", ["correlation_id", "reply_to"])
@pytest.mark.parametrize("value", [None, ""])
def test_missing_routing_metadata_dead_letters(field, value):
    message, channel, events = resources(**{field: value})
    asyncio.run(consumer().handle(message, channel))
    assert events == ["reject"]
    message.reject.assert_awaited_once_with(requeue=False)
    channel.default_exchange.publish.assert_not_called()


def test_transient_publish_failure_retries_reply_without_reexecution(monkeypatch):
    monkeypatch.setattr(
        module.analysis_service, "run_analysis", AsyncMock(return_value="{}")
    )
    message, channel, _ = resources()
    channel.default_exchange.publish.side_effect = [TimeoutError(), None]
    runtime = consumer()
    runtime._retry = AsyncMock()
    asyncio.run(runtime.handle(message, channel))
    message.ack.assert_awaited_once()
    message.reject.assert_not_called()
    message.channel.close.assert_not_called()
    channel.close.assert_not_called()
    assert channel.default_exchange.publish.await_count == 2
    module.analysis_service.run_analysis.assert_awaited_once()


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


def test_missing_destination_is_rejected_before_execution(monkeypatch):
    execute = AsyncMock()
    monkeypatch.setattr(module.analysis_service, "run_analysis", execute)
    message, connection, _ = resources()
    connection.reply_channel.declare_queue.side_effect = ChannelNotFoundEntity(
        404, "NOT_FOUND"
    )
    asyncio.run(consumer().handle(message, connection))
    execute.assert_not_called()
    connection.reply_channel.declare_queue.assert_awaited_once_with(
        "replies", passive=True, timeout=5
    )
    message.reject.assert_awaited_once_with(requeue=False)
    message.channel.close.assert_not_called()


def test_broker_nack_does_not_interrupt_another_delivery(monkeypatch):
    async def scenario():
        running, release = asyncio.Event(), asyncio.Event()

        async def execute(request, **kwargs):
            if request.request_id == "other-request":
                running.set()
                await release.wait()
            return "{}"

        mock = AsyncMock(side_effect=execute)
        monkeypatch.setattr(module.analysis_service, "run_analysis", mock)
        runtime = consumer()
        bad, connection, _ = resources()
        good, other, _ = resources(correlation_id="other-request")
        # Both source deliveries share the same consuming channel/connection.
        good.channel = bad.channel
        connection.channel.side_effect = [other.reply_channel, connection.reply_channel]
        connection.default_exchange.publish.side_effect = DeliveryError(
            None, Basic.Nack(delivery_tag=1)
        )
        first = asyncio.create_task(runtime.handle(good, connection))
        await running.wait()
        await runtime.handle(bad, connection)
        bad.reject.assert_awaited_once_with(requeue=False)
        assert not first.done()
        bad.channel.close.assert_not_called()
        connection.close.assert_not_called()
        release.set()
        await first
        good.ack.assert_awaited_once()
        assert sorted(call.args[0].request_id for call in mock.await_args_list) == [
            "other-request",
            "request-1",
        ]

    asyncio.run(scenario())


@pytest.mark.parametrize("failure_stage", ["publish", "ack"])
def test_redelivery_reuses_completed_result_after_handler_exits(
    monkeypatch, failure_stage
):
    async def scenario():
        execute = AsyncMock(return_value='{"value": 42}')
        monkeypatch.setattr(module.analysis_service, "run_analysis", execute)
        runtime = consumer()
        first, connection, _ = resources()

        async def disconnected(*args, **kwargs):
            connection.is_closed = True
            raise ConnectionError("disconnected")

        if failure_stage == "publish":
            connection.default_exchange.publish.side_effect = disconnected
        else:
            first.ack.side_effect = disconnected
        await runtime.handle(first, connection)
        assert not runtime.executions
        assert runtime.completed.get("request-1") is not None
        second, replacement, _ = resources(redelivered=True)
        await runtime.handle(second, replacement)
        execute.assert_awaited_once()
        assert json.loads(replacement.default_exchange.publish.call_args.args[0].body)[
            "result"
        ] == {"value": 42}
        second.ack.assert_awaited_once()
        assert runtime.completed.get("request-1") is None

    asyncio.run(scenario())


def test_settled_request_id_can_start_a_fresh_execution(monkeypatch):
    async def scenario():
        execute = AsyncMock(return_value="{}")
        monkeypatch.setattr(module.analysis_service, "run_analysis", execute)
        runtime = consumer()
        for _ in range(2):
            message, connection, _ = resources()
            await runtime.handle(message, connection)
        assert execute.await_count == 2

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "bound", ["REPLY_CACHE_TTL", "REPLY_CACHE_ENTRIES", "REPLY_CACHE_BYTES"]
)
def test_expired_or_evicted_reply_allows_recomputation(monkeypatch, bound):
    async def scenario():
        monkeypatch.setattr(module, bound, 0)
        execute = AsyncMock(return_value="{}")
        monkeypatch.setattr(module.analysis_service, "run_analysis", execute)
        runtime = consumer()
        first, connection, _ = resources()

        async def disconnected(*args, **kwargs):
            connection.is_closed = True
            raise ConnectionError()

        connection.default_exchange.publish.side_effect = disconnected
        await runtime.handle(first, connection)
        second, replacement, _ = resources(redelivered=True)
        await runtime.handle(second, replacement)
        assert execute.await_count == 2
        second.ack.assert_awaited_once()

    asyncio.run(scenario())


def test_cancelled_handler_retains_its_finishing_execution_for_redelivery(monkeypatch):
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        async def execute(*args, **kwargs):
            entered.set()
            await release.wait()
            return "{}"

        mock = AsyncMock(side_effect=execute)
        monkeypatch.setattr(module.analysis_service, "run_analysis", mock)
        runtime = consumer()
        first, connection, _ = resources()
        handler = asyncio.create_task(runtime.handle(first, connection))
        await asyncio.wait_for(entered.wait(), 1)
        execution = runtime.executions["request-1"]
        handler.cancel()
        with pytest.raises(asyncio.CancelledError):
            await handler
        assert not execution.cancelled()
        release.set()
        await execution
        assert not runtime.executions
        second, replacement, _ = resources(redelivered=True)
        await runtime.handle(second, replacement)
        mock.assert_awaited_once()
        second.ack.assert_awaited_once()

    asyncio.run(scenario())


def test_reply_cache_expires_and_evicts_by_bytes_and_count(monkeypatch):
    now = [0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(module, "REPLY_CACHE_ENTRIES", 2)
    monkeypatch.setattr(module, "REPLY_CACHE_BYTES", 5)
    cache = module.ReplyCache()
    cache.put("first", b"12")
    cache.put("second", b"34")
    cache.put("third", b"56")
    assert cache.get("first") is None
    assert cache.get("second") == b"34"
    cache.put("fourth", b"7890")
    assert cache.size == 4
    assert cache.get("second") is None
    cache.put("oversize", b"123456")
    assert cache.get("oversize") is None
    now[0] = module.REPLY_CACHE_TTL
    assert cache.get("fourth") is None
    assert cache.size == 0


def test_diagnostics_include_channel_reason_but_never_bodies_or_urls():
    reason = module.error_summary(
        ExceptionGroup(
            "connection",
            [
                ChannelPreconditionFailed(
                    406,
                    "PRECONDITION_FAILED inequivalent arg x-max-length\namqp://user:secret@host",
                )
            ],
        )
    )
    assert "406" in reason and "inequivalent arg" in reason
    assert "secret" not in reason and "\n" not in reason
    assert "secret" not in module.error_summary(
        ConnectionError("amqp://user:secret@host")
    )
    assert "secret" not in module.error_summary(
        DeliveryError(SimpleNamespace(body=b"secret"), Basic.Nack())
    )


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


def test_drain_interrupts_reply_retry_without_acknowledging(monkeypatch):
    async def scenario():
        monkeypatch.setattr(
            module.analysis_service, "run_analysis", AsyncMock(return_value="{}")
        )
        message, connection, _ = resources()
        connection.default_exchange.publish.side_effect = TimeoutError()
        runtime = consumer(shutdown_drain_timeout=1)
        retrying = asyncio.Event()
        original_retry = runtime._retry

        async def retry(delay):
            retrying.set()
            await original_retry(delay)

        runtime._retry = retry
        delivery = asyncio.create_task(runtime.handle(message, connection))
        runtime.deliveries.add(delivery)
        await asyncio.wait_for(retrying.wait(), 1)
        await asyncio.wait_for(runtime.drain(), 1)
        await delivery
        message.ack.assert_not_called()
        message.reject.assert_not_called()
        message.channel.close.assert_not_called()
        await runtime.close()

    asyncio.run(scenario())


def test_close_cleans_up_reply_channels_concurrently(monkeypatch):
    async def scenario():
        runtime = consumer(operation_timeout=1)
        entered, closing, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        started_count, closing_count = 0, 0

        async def execute(*args, **kwargs):
            nonlocal started_count
            started_count += 1
            if started_count == 3:
                entered.set()
            await asyncio.Event().wait()

        async def close():
            nonlocal closing_count
            closing_count += 1
            if closing_count == 3:
                closing.set()
            await release.wait()

        monkeypatch.setattr(module.analysis_service, "run_analysis", execute)
        for index in range(3):
            message, connection, _ = resources(correlation_id=f"job-{index}")
            connection.reply_channel.close.side_effect = close
            runtime.deliveries.add(
                asyncio.create_task(runtime.handle(message, connection))
            )
        await asyncio.wait_for(entered.wait(), 1)
        shutdown = asyncio.create_task(runtime.close())
        try:
            await asyncio.wait_for(closing.wait(), 1)
        finally:
            release.set()
        await asyncio.wait_for(shutdown, 1)
        assert all(task.done() for task in runtime.deliveries)

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

            def __init__(self):
                self.opened = 0

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                await self.close()

            async def channel(self, **kwargs):
                self.opened += 1
                return channel if self.opened == 1 else resources()[1].reply_channel

            async def close(self):
                self.is_closed = True

        connection = Connection()
        monkeypatch.setattr(
            module.aio_pika, "connect", AsyncMock(return_value=connection)
        )
        runtime = consumer(enabled=True, url="amqp://localhost")
        runtime.start()
        await asyncio.wait_for(started.wait(), 2)
        channel.set_qos.assert_awaited_once_with(prefetch_count=2)
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
