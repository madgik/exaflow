from contextlib import asynccontextmanager

from fastapi import FastAPI

from exaflow.controller import config as ctrl_config
from exaflow.controller.http import endpoints
from exaflow.controller.http.error_handlers import register_error_handlers
from exaflow.controller.http.request_body_limit import RequestBodyLimitMiddleware
from exaflow.controller.services import get_worker_landscape_aggregator
from exaflow.controller.services.rabbitmq.consumer import AnalysisConsumer
from exaflow.controller.services.rabbitmq.settings import RabbitMQSettings


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Validate configuration before starting resources, without connecting to AMQP.
    settings = RabbitMQSettings.model_validate(ctrl_config.get("rabbitmq", {}))
    await endpoints.startup()
    consumer = AnalysisConsumer(
        settings,
        lambda: get_worker_landscape_aggregator().initial_refresh_complete.is_set(),
    )
    _app.state.rabbitmq = consumer
    try:
        consumer.start()
        yield
    finally:
        try:
            await consumer.drain()
        finally:
            try:
                await endpoints.shutdown()
            finally:
                await consumer.close()


app = FastAPI(
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)
app.add_middleware(RequestBodyLimitMiddleware)
register_error_handlers(app)
app.include_router(endpoints.router)
