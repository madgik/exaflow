"""Bounded AMQP delivery handling; the broker owns unfinished requests."""

import asyncio
import json
import logging
import re
import time
import traceback
from collections import OrderedDict
from collections.abc import Callable

import aio_pika
from pydantic import BaseModel
from pydantic import TypeAdapter
from pydantic import ValidationError

from exaflow.controller import BACKGROUND_LOGGER_NAME
from exaflow.controller.services import analysis_service
from exaflow.controller.services.analysis_execution import max_concurrent_experiments
from exaflow.controller.services.api.analysis_request_dtos import AnalysisRequestDTO
from exaflow.controller.services.api.analysis_request_dtos import RequestID
from exaflow.controller.services.api.analysis_request_validator import BadRequest
from exaflow.controller.services.api.limits import (
    MAX_REQUEST_BODY_BYTES as MAX_REQUEST_BYTES,
)
from exaflow.controller.services.errors import ControllerShuttingDownError
from exaflow.controller.services.errors import WorkerTaskTimeoutError
from exaflow.controller.services.errors import WorkerUnresponsiveError
from exaflow.controller.services.rabbitmq.settings import RabbitMQSettings
from exaflow.controller.worker_client.app import WorkerMessageTooLargeError
from exaflow.data_filters import FilterError
from exaflow.logging_utils import safe_log_value
from exaflow.worker_communication import BadUserInput
from exaflow.worker_communication import DataModelUnavailable
from exaflow.worker_communication import DatasetUnavailable
from exaflow.worker_communication import InsufficientDataError

LOGGER = logging.getLogger(BACKGROUND_LOGGER_NAME)
REQUEST_ID_ADAPTER = TypeAdapter(RequestID)
REPLY_CACHE_TTL = 300
REPLY_CACHE_ENTRIES = 128
REPLY_CACHE_BYTES = 64 * 1024 * 1024


def error_summary(exc: Exception) -> str:
    if isinstance(exc, BaseExceptionGroup):
        return error_summary(exc.exceptions[0])
    summary = type(exc).__name__
    if isinstance(exc, aio_pika.exceptions.ChannelClosed):
        # Only broker channel errors carry diagnostic text. DeliveryError includes
        # the published body, and connection errors can contain the AMQP URL.
        details = " ".join(str(arg) for arg in exc.args)
        details = re.sub(r"amqps?://\S+", "<redacted>", details)
        summary += ": " + safe_log_value(details)[:512]
    return summary


class ReplyCache:
    """Best-effort retention of bounded replies for unsettled deliveries."""

    def __init__(self):
        self.entries = OrderedDict()
        self.size = 0

    def discard(self, request_id):
        entry = self.entries.pop(request_id, None)
        if entry is not None:
            self.size -= len(entry[1])

    def prune(self):
        now = time.monotonic()
        for request_id, (expires, _) in list(self.entries.items()):
            if expires <= now:
                self.discard(request_id)

    def get(self, request_id):
        self.prune()
        entry = self.entries.get(request_id)
        return None if entry is None else entry[1]

    def put(self, request_id, body):
        self.prune()
        self.discard(request_id)
        if len(body) > REPLY_CACHE_BYTES:
            return
        self.entries[request_id] = (time.monotonic() + REPLY_CACHE_TTL, body)
        self.size += len(body)
        while len(self.entries) > REPLY_CACHE_ENTRIES or self.size > REPLY_CACHE_BYTES:
            self.discard(next(iter(self.entries)))

    def clear(self):
        self.entries.clear()
        self.size = 0


class FailureDetail(BaseModel):
    code: str
    message: str


class AnalysisReply(BaseModel):
    status: str
    result: object | None = None
    error: FailureDetail | None = None


def failure(code: str, message: str) -> AnalysisReply:
    return AnalysisReply(
        status="failed", error=FailureDetail(code=code, message=message)
    )


class AnalysisConsumer:
    def __init__(self, settings: RabbitMQSettings, landscape_ready: Callable[[], bool]):
        self.settings = settings
        self.landscape_ready = landscape_ready
        self.connection = None
        self.publish_connection = None
        self._publish_lock = asyncio.Lock()
        self.runner = None
        self.iterator = None
        self.finish = asyncio.Event()
        self.retry_stop = asyncio.Event()
        self.consuming = False
        self.stopping = False
        self.last_error = None
        self.deliveries: set[asyncio.Task] = set()
        self.executions: dict[str, asyncio.Task] = {}
        self.completed = ReplyCache()
        self.limit = max_concurrent_experiments()
        self.handlers = asyncio.Semaphore(self.limit)

    def start(self):
        if self.settings.enabled:
            self.runner = asyncio.create_task(self._run(), name="rabbitmq-consumer")

    def health(self) -> dict:
        # Same keys as /healthcheck/rabbitmq. connected covers both AMQP
        # connections: a blocked or dropped publisher cannot complete work.
        return {
            "enabled": self.settings.enabled,
            "connected": self._amqp_open(self.connection)
            and self._amqp_open(self.publish_connection),
            "consuming": self.consuming,
            "active_jobs": sum(not task.done() for task in self.executions.values()),
            "last_error": self.last_error,
        }

    def _amqp_open(self, connection):
        if connection is None:
            return False
        try:
            if connection.is_closed:
                return False
            connected = getattr(connection, "connected", None)
            if isinstance(connected, asyncio.Event) and not connected.is_set():
                return False
            if hasattr(connection, "transport"):
                transport = connection.transport
                if transport is None:
                    return False
                underlay = getattr(transport, "connection", None)
                if underlay is not None and underlay.is_closed:
                    return False
            return True
        except Exception:
            return False

    def _publish_connection_failed(self, exc, used_connection=None):
        target = (
            used_connection if used_connection is not None else self.publish_connection
        )
        if not self._amqp_open(target):
            return True
        if isinstance(exc, BaseExceptionGroup) and exc.exceptions:
            exc = exc.exceptions[0]
        # ConnectionClosed is a ConnectionError. Channel and timeout errors are not.
        return isinstance(exc, ConnectionError)

    async def _close_amqp(self, connection):
        if connection is None:
            return
        try:
            if connection.is_closed:
                return
            await asyncio.wait_for(connection.close(), self.settings.operation_timeout)
        except Exception as exc:
            # The exception text can include the AMQP URL.
            LOGGER.warning("RabbitMQ connection close failed (%s)", type(exc).__name__)

    async def _ensure_publish_connection(self):
        # Memory and disk alarms block publishers. Acknowledgements stay on
        # self.connection so connection.blocked cannot stall basic.ack.
        async with self._publish_lock:
            if self.stopping:
                raise ConnectionError("RabbitMQ publisher is stopping")
            if self._amqp_open(self.publish_connection):
                return self.publish_connection
            stale = self.publish_connection
            self.publish_connection = None
            await self._close_amqp(stale)
            if self.stopping:
                raise ConnectionError("RabbitMQ publisher is stopping")
            connection = await asyncio.wait_for(
                aio_pika.connect(
                    self.settings.url,
                    timeout=self.settings.operation_timeout,
                    heartbeat=30,
                    client_properties={"connection_name": "exaflow-analysis-publisher"},
                ),
                self.settings.operation_timeout,
            )
            try:
                if self.stopping:
                    raise ConnectionError("RabbitMQ publisher is stopping")
                self.publish_connection = connection
            except BaseException:
                await self._close_amqp(connection)
                raise
            return connection

    async def _retire_publish_connection(self, failed):
        if failed is None:
            return
        async with self._publish_lock:
            if self.publish_connection is not failed:
                return
            self.publish_connection = None
            await self._close_amqp(failed)

    async def _open_reply_channel(self, connection):
        return await asyncio.wait_for(
            connection.channel(
                publisher_confirms=True,
                on_return_raises=True,
            ),
            self.settings.operation_timeout,
        )

    async def _run(self):
        delay = 1
        while not self.stopping:
            try:
                while not self.stopping and not self.landscape_ready():
                    await asyncio.sleep(0.1)
                if self.stopping:
                    return
                self.connection = await aio_pika.connect(
                    self.settings.url,
                    timeout=self.settings.operation_timeout,
                    heartbeat=30,
                    client_properties={"connection_name": "exaflow-analysis-consumer"},
                )
                async with self.connection:
                    await self._ensure_publish_connection()
                    async with asyncio.timeout(self.settings.operation_timeout):
                        channel = await self.connection.channel(
                            publisher_confirms=True,
                            on_return_raises=True,
                        )
                        await channel.set_qos(prefetch_count=self.limit)
                        await channel.declare_queue(
                            self.settings.dead_letter_queue,
                            durable=True,
                            arguments={
                                "x-queue-type": "quorum",
                                "x-message-ttl": self.settings.dead_letter_ttl_ms,
                                "x-max-length-bytes": self.settings.queue_max_bytes,
                                "x-overflow": "reject-publish",
                            },
                        )
                        queue = await channel.declare_queue(
                            self.settings.request_queue,
                            durable=True,
                            arguments={
                                "x-queue-type": "quorum",
                                "x-dead-letter-exchange": "",
                                "x-dead-letter-routing-key": self.settings.dead_letter_queue,
                                "x-dead-letter-strategy": "at-least-once",
                                "x-max-length-bytes": self.settings.queue_max_bytes,
                                "x-overflow": "reject-publish",
                            },
                        )
                    self.iterator = queue.iterator()
                    async with self.iterator as messages:
                        self.consuming = True
                        self.last_error = None
                        delay = 1
                        LOGGER.info("RabbitMQ consumer connected")
                        async with asyncio.TaskGroup() as tasks:
                            monitor = tasks.create_task(
                                self._monitor_subscription(queue)
                            )
                            try:
                                async for message in messages:
                                    if self.stopping:
                                        break
                                    await self.handlers.acquire()
                                    if self.stopping:
                                        self.handlers.release()
                                        break
                                    task = asyncio.create_task(
                                        self.handle(message, self.connection)
                                    )
                                    self.deliveries.add(task)
                                    task.add_done_callback(self._delivery_done)
                            finally:
                                monitor.cancel()
                    if self.stopping:
                        await self.finish.wait()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Exception text can include credentials or broker message bodies.
                self.last_error = type(exc).__name__
                LOGGER.warning(
                    "RabbitMQ connection unavailable (%s)", error_summary(exc)
                )
            finally:
                self.consuming = False
                self.iterator = None
                self.completed.prune()
            if not self.stopping:
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30)

    async def _monitor_subscription(self, queue):
        # aiormq consumes Basic.Cancel without waking QueueIterator. This process
        # owns the sole subscription, so zero consumers means it must resubscribe.
        while not self.stopping:
            await asyncio.sleep(1)
            self.completed.prune()
            if self.stopping:
                return
            try:
                declaration = await queue.declare(
                    timeout=self.settings.operation_timeout
                )
                if declaration.consumer_count == 0:
                    raise ConnectionError("RabbitMQ subscription cancelled")
            except Exception:
                self.consuming = False
                raise
            if not self._amqp_open(self.publish_connection):
                try:
                    await self._ensure_publish_connection()
                except Exception as exc:
                    self.last_error = type(exc).__name__
                    LOGGER.warning(
                        "RabbitMQ publisher connection unavailable (%s)",
                        error_summary(exc),
                    )

    def _delivery_done(self, task):
        self.deliveries.discard(task)
        self.handlers.release()
        if not task.cancelled() and task.exception() is not None:
            LOGGER.error(
                "RabbitMQ delivery task failed (%s)", type(task.exception()).__name__
            )

    async def _execute(self, request: AnalysisRequestDTO) -> AnalysisReply:
        try:
            result = await analysis_service.run_analysis(
                request, wait_for_capacity=True
            )
            return AnalysisReply(status="succeeded", result=json.loads(result))
        except ControllerShuttingDownError:
            raise
        except InsufficientDataError:
            return failure("insufficient_data", "Insufficient data for this analysis.")
        except (
            BadRequest,
            BadUserInput,
            DataModelUnavailable,
            DatasetUnavailable,
            FilterError,
            WorkerTaskTimeoutError,
            WorkerUnresponsiveError,
            WorkerMessageTooLargeError,
        ) as exc:
            return failure(type(exc).__name__, getattr(exc, "message", str(exc)))
        except Exception as exc:
            LOGGER.error(
                "Analysis failed request_id=%s (%s)\n%s",
                safe_log_value(request.request_id),
                type(exc).__name__,
                "".join(traceback.format_tb(exc.__traceback__)),
            )
            return failure("internal_error", "Analysis execution failed.")

    def _encode_reply(self, reply):
        body = reply.model_dump_json(exclude_none=True).encode()
        if len(body) > self.settings.max_result_bytes:
            body = (
                failure(
                    "result_too_large", "Result exceeds the configured message limit."
                )
                .model_dump_json(exclude_none=True)
                .encode()
            )
        return body

    async def _execute_reply(self, request):
        body = self._encode_reply(await self._execute(request))
        if not self.finish.is_set():
            self.completed.put(request.request_id, body)
        return body

    def _request_reply(self, message):
        if len(message.body) > MAX_REQUEST_BYTES:
            return None, self._encode_reply(
                failure("request_too_large", "Request exceeds 16 MiB.")
            )
        try:
            request_id = REQUEST_ID_ADAPTER.validate_python(message.correlation_id)
            request = AnalysisRequestDTO.model_validate_json(message.body)
            if request.request_id is not None and request.request_id != request_id:
                raise ValueError("Conflicting request identifiers")
            request.request_id = request_id
        except (ValidationError, ValueError):
            return None, self._encode_reply(
                failure(
                    "invalid_request",
                    "Request does not match the analysis schema or identifiers.",
                )
            )
        execution = self.executions.get(request_id)
        if execution is None and message.redelivered:
            body = self.completed.get(request_id)
            if body is not None:
                return None, body
        if execution is None:
            execution = asyncio.create_task(
                self._execute_reply(request), name="queued-analysis"
            )
            self.executions[request_id] = execution
        return execution, None

    def _delivery_available(self, message, connection):
        try:
            return self._amqp_open(connection) and not message.channel.is_closed
        except aio_pika.exceptions.ChannelInvalidStateError:
            return False

    async def _close_reply_channel(self, channel):
        if channel is not None:
            try:
                await asyncio.wait_for(channel.close(), self.settings.operation_timeout)
            except Exception as exc:
                LOGGER.warning(
                    "RabbitMQ reply channel close failed (%s)", error_summary(exc)
                )

    async def _retry(self, delay):
        try:
            await asyncio.wait_for(self.retry_stop.wait(), delay)
        except TimeoutError:
            pass

    async def _settle(self, message, request_id, *, reject=False):
        try:
            await asyncio.wait_for(
                message.reject(requeue=False) if reject else message.ack(),
                self.settings.operation_timeout,
            )
        except Exception as exc:
            LOGGER.warning("RabbitMQ settlement interrupted (%s)", error_summary(exc))
            # A failed consumer-channel write is an actual transport failure.
            # Retain completed replies while recovery requeues its deliveries.
            try:
                await asyncio.wait_for(
                    message.channel.close(), self.settings.operation_timeout
                )
            except Exception as close_error:
                LOGGER.warning(
                    "RabbitMQ delivery channel close failed (%s)",
                    error_summary(close_error),
                )
        else:
            self.completed.discard(request_id)

    async def handle(self, message, connection):
        request_id = message.correlation_id
        execution = None
        publish_connection = None
        channel = None
        body = None
        delay = 1
        try:
            if not request_id or not message.reply_to:
                await self._settle(message, request_id, reject=True)
                return
            while self._delivery_available(message, connection):
                try:
                    publish_connection = await self._ensure_publish_connection()
                    channel = await self._open_reply_channel(publish_connection)
                    # A missing queue closes this isolated reply channel only.
                    await channel.declare_queue(
                        message.reply_to,
                        passive=True,
                        timeout=self.settings.operation_timeout,
                    )
                    if body is None:
                        execution, body = self._request_reply(message)
                        if execution is not None:
                            body = await asyncio.shield(execution)
                    if not self._delivery_available(message, connection):
                        return
                    await channel.default_exchange.publish(
                        aio_pika.Message(
                            body=body,
                            content_type="application/json",
                            correlation_id=request_id,
                            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                            expiration=self.settings.reply_ttl_ms / 1000,
                        ),
                        routing_key=message.reply_to,
                        mandatory=True,
                        timeout=self.settings.publish_timeout,
                    )
                except aio_pika.exceptions.ChannelNotFoundEntity as exc:
                    LOGGER.warning(
                        "RabbitMQ reply destination unavailable request_id=%s (%s)",
                        safe_log_value(request_id),
                        error_summary(exc),
                    )
                    await self._settle(message, request_id, reject=True)
                    return
                except aio_pika.exceptions.DeliveryError as exc:
                    # Both mandatory returns and broker nacks are terminal for
                    # this request; neither invalidates the consuming channel.
                    LOGGER.warning(
                        "RabbitMQ reply rejected request_id=%s (%s)",
                        safe_log_value(request_id),
                        type(exc).__name__,
                    )
                    await self._settle(message, request_id, reject=True)
                    return
                except ControllerShuttingDownError:
                    return
                except Exception as exc:
                    self.last_error = type(exc).__name__
                    LOGGER.warning(
                        "RabbitMQ delivery interrupted request_id=%s (%s)",
                        safe_log_value(request_id),
                        error_summary(exc),
                    )
                    await self._close_reply_channel(channel)
                    channel = None
                    # A dropped publisher is replaced in place. Completed replies
                    # stay cached, and the consuming connection keeps its deliveries.
                    if (
                        publish_connection is not None
                        and self._publish_connection_failed(exc, publish_connection)
                    ):
                        await self._retire_publish_connection(publish_connection)
                    publish_connection = None
                    if self.stopping or not self._delivery_available(
                        message, connection
                    ):
                        return
                    await self._retry(delay)
                    if self.stopping:
                        return
                    delay = min(delay * 2, 30)
                else:
                    await self._settle(message, request_id)
                    return
        except asyncio.CancelledError:
            raise
        finally:
            if execution is not None:

                def release(task):
                    if self.executions.get(request_id) is task:
                        self.executions.pop(request_id)
                    if not task.cancelled():
                        task.exception()

                if execution.done():
                    release(execution)
                else:
                    execution.add_done_callback(release)
            await self._close_reply_channel(channel)

    async def drain(self):
        self.stopping = True
        self.retry_stop.set()
        self.consuming = False
        if self.iterator is not None:
            try:
                await asyncio.wait_for(
                    self.iterator.close(), self.settings.operation_timeout
                )
            except Exception as exc:
                LOGGER.warning("RabbitMQ intake stop failed (%s)", type(exc).__name__)
        if self.deliveries:
            await asyncio.wait(
                self.deliveries, timeout=self.settings.shutdown_drain_timeout
            )

    async def close(self):
        self.stopping = True
        self.retry_stop.set()
        self.finish.set()
        tasks = set(self.deliveries) | set(self.executions.values())
        if self.runner is not None:
            tasks.add(self.runner)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait(tasks, timeout=self.settings.operation_timeout)
        # One operation_timeout budget for both closes; gather keeps them concurrent.
        consumer_connection = self.connection
        publisher_connection = self.publish_connection
        await asyncio.gather(
            self._close_amqp(consumer_connection),
            self._close_amqp(publisher_connection),
        )
        self.completed.clear()
