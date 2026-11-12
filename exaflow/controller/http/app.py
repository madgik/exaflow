from contextlib import asynccontextmanager

from fastapi import FastAPI

from exaflow.controller.http import endpoints
from exaflow.controller.http.error_handlers import register_error_handlers
from exaflow.controller.http.request_body_limit import RequestBodyLimitMiddleware


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await endpoints.startup()
    try:
        yield
    finally:
        await endpoints.shutdown()


app = FastAPI(
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)
app.add_middleware(RequestBodyLimitMiddleware)
register_error_handlers(app)
app.include_router(endpoints.router)
