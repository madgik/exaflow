from contextlib import asynccontextmanager

from fastapi import FastAPI

from exaflow.controller.http import endpoints
from exaflow.controller.http.error_handlers import register_error_handlers


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
register_error_handlers(app)
app.include_router(endpoints.router)


def test_client():
    """Quart-shaped ASGI client. httpx is a dev dependency, so import it here."""
    from exaflow.controller.http.asgi_client import ControllerClient

    return ControllerClient(app)


app.test_client = test_client
