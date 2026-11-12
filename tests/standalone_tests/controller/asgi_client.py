"""httpx client that calls the controller app in-process, without a server."""

import httpx


def controller_client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://controller"
    )
