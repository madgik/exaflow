"""Generated schema routes stay off. The HTTP contract is the API."""

import asyncio

import httpx
from exaflow.controller.http.app import app


def test_generated_docs_routes_are_disabled_and_return_404():
    assert app.docs_url is None
    assert app.redoc_url is None
    assert app.openapi_url is None

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://controller"
        ) as client:
            return {
                path: (await client.get(path)).status_code
                for path in ("/docs", "/redoc", "/openapi.json")
            }

    assert asyncio.run(scenario()) == {
        "/docs": 404,
        "/redoc": 404,
        "/openapi.json": 404,
    }
