"""Test client matching the Quart methods the controller tests call."""

import httpx


class _Response:
    def __init__(self, response: httpx.Response):
        self.status_code = response.status_code
        self.content_type = response.headers.get("content-type", "")
        self._response = response

    async def get_data(self, as_text: bool = False):
        if as_text:
            return self._response.text
        return self._response.content


class ControllerClient:
    def __init__(self, app):
        self._app = app
        self._client: httpx.AsyncClient | None = None
        self._managed = False

    async def __aenter__(self):
        await self._ensure()
        self._managed = True
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self._close()
        return False

    async def get(self, path, headers=None):
        return await self._send("get", path, headers=headers)

    async def post(self, path, json=None, headers=None, data=None):
        return await self._send("post", path, headers=headers, json=json, data=data)

    async def _ensure(self) -> None:
        if self._client is not None:
            return
        transport = httpx.ASGITransport(app=self._app)
        self._client = httpx.AsyncClient(
            transport=transport, base_url="http://controller"
        )
        await self._client.__aenter__()

    async def _close(self) -> None:
        client = self._client
        self._client = None
        self._managed = False
        if client is not None:
            await client.__aexit__(None, None, None)

    async def _send(self, method, path, headers=None, json=None, data=None):
        close_after = self._client is None and not self._managed
        await self._ensure()
        try:
            kwargs = {"headers": headers}
            if method == "post":
                if json is not None:
                    kwargs["json"] = json
                elif data is not None:
                    kwargs["content"] = data
            response = await getattr(self._client, method)(path, **kwargs)
            return _Response(response)
        finally:
            if close_after:
                await self._close()
