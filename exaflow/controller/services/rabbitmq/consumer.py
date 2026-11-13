"""Bounded AMQP delivery handling; the broker owns unfinished requests."""

import asyncio
import json
import logging
from collections.abc import Callable

import aio_pika
from pydantic import BaseModel
from pydantic import ValidationError

from exaflow.controller import BACKGROUND_LOGGER_NAME
from exaflow.controller.services import analysis_service
from exaflow.controller.services.analysis_execution import max_concurrent_experiments
from exaflow.controller.services.api.analysis_request_dtos import AnalysisRequestDTO
from exaflow.controller.services.api.analysis_request_validator import BadRequest
from exaflow.controller.services.errors import ControllerShuttingDownError
from exaflow.controller.services.rabbitmq.settings import RabbitMQSettings
from exaflow.data_filters import FilterError
from exaflow.smpc_cluster_communication import SMPCUsageError
from exaflow.worker_communication import BadUserInput
from exaflow.worker_communication import DataModelUnavailable
from exaflow.worker_communication import DatasetUnavailable
from exaflow.worker_communication import InsufficientDataError

LOGGER = logging.getLogger(BACKGROUND_LOGGER_NAME)
MAX_REQUEST_BYTES = 16 * 1024 * 1024


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
        self.runner = None
        self.iterator = None
        self.finish = asyncio.Event()
        self.consuming = False
        self.stopping = False
        self.last_error = None
        self.deliveries: set[asyncio.Task] = set()
        self.executions: dict[str, asyncio.Task] = {}
        self.limit = max_concurrent_experiments()
        self.handlers = asyncio.Semaphore(self.limit)

    def start(self):
        if self.settings.enabled:
            self.runner = asyncio.create_task(self._run(), name="rabbitmq-consumer")

    def health(self) -> dict:
        return {
            "enabled": self.settings.enabled,
            "connected": self.connection is not None and not self.connection.is_closed,
            "consuming": self.consuming,
            "active_jobs": len(self.executions),
            "last_error": self.last_error,
        }

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
                                        self.handle(message, channel)
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
                LOGGER.warning("RabbitMQ connection unavailable (%s)", self.last_error)
            finally:
                self.consuming = False
                self.iterator = None
            if not self.stopping:
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30)

    async def _monitor_subscription(self, queue):
        # aiormq consumes Basic.Cancel without waking QueueIterator. This process
        # owns the sole subscription, so zero consumers means it must resubscribe.
        while not self.stopping:
            await asyncio.sleep(1)
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
            SMPCUsageError,
        ) as exc:
            return failure(type(exc).__name__, getattr(exc, "message", str(exc)))
        except Exception as exc:
            LOGGER.error(
                "Analysis failed request_id=%s (%s)",
                request.request_id,
                type(exc).__name__,
            )
            return failure("internal_error", "Analysis execution failed.")

    async def handle(self, message, channel):
        execution = None
        request_id = message.correlation_id
        try:
            if not request_id or not message.reply_to:
                await message.reject(requeue=False)
                return
            if len(message.body) > MAX_REQUEST_BYTES:
                reply = failure("request_too_large", "Request exceeds 16 MiB.")
            else:
                try:
                    request = AnalysisRequestDTO.model_validate_json(message.body)
                    if (
                        request.request_id is not None
                        and request.request_id != request_id
                    ):
                        raise ValueError("Conflicting request identifiers")
                    request.request_id = request_id
                except (ValidationError, ValueError):
                    reply = failure(
                        "invalid_request",
                        "Request does not match the analysis schema or identifiers.",
                    )
                else:
                    execution = self.executions.get(request_id)
                    if execution is None:
                        execution = asyncio.create_task(
                            self._execute(request), name="queued-analysis"
                        )
                        self.executions[request_id] = execution
                    reply = await asyncio.shield(execution)
            body = reply.model_dump_json(exclude_none=True).encode()
            if len(body) > self.settings.max_result_bytes:
                body = (
                    failure(
                        "result_too_large",
                        "Result exceeds the configured message limit.",
                    )
                    .model_dump_json(exclude_none=True)
                    .encode()
                )
            try:
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
                    timeout=self.settings.operation_timeout,
                )
            except aio_pika.exceptions.PublishError:
                # A returned mandatory reply has no destination. Preserve the request in DLQ.
                await message.reject(requeue=False)
                return
            await message.ack()
        except asyncio.CancelledError:
            raise
        except ControllerShuttingDownError:
            # The request remains unacknowledged for another controller attempt.
            pass
        except Exception as exc:
            self.last_error = type(exc).__name__
            LOGGER.warning(
                "RabbitMQ delivery interrupted request_id=%s (%s)",
                request_id,
                self.last_error,
            )
            # Closing the delivery channel requeues unfinished messages. Never ack an
            # old delivery on a replacement channel after reconnecting.
            try:
                await asyncio.wait_for(channel.close(), self.settings.operation_timeout)
            except Exception as close_error:
                LOGGER.warning(
                    "RabbitMQ channel close failed (%s)", type(close_error).__name__
                )
        finally:
            if execution is not None and execution.done():
                if self.executions.get(request_id) is execution:
                    self.executions.pop(request_id)
                if not execution.cancelled():
                    execution.exception()

    async def drain(self):
        self.stopping = True
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
        self.finish.set()
        tasks = set(self.deliveries) | set(self.executions.values())
        if self.runner is not None:
            tasks.add(self.runner)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait(tasks, timeout=self.settings.operation_timeout)
        if self.connection is not None and not self.connection.is_closed:
            try:
                await asyncio.wait_for(
                    self.connection.close(), self.settings.operation_timeout
                )
            except Exception as exc:
                LOGGER.warning(
                    "RabbitMQ connection close failed (%s)", type(exc).__name__
                )
