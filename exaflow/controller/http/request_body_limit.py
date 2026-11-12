"""Reject a request body larger than ``MAX_REQUEST_BODY_BYTES`` with 413."""

from fastapi import HTTPException

# Uvicorn does not limit the request body. 16 MiB is the limit Quart applied.
MAX_REQUEST_BODY_BYTES = 16 * 1024 * 1024


class RequestBodyTooLargeError(HTTPException):
    # An HTTPException, because FastAPI answers 400 for any other error raised
    # while it reads the body.
    def __init__(self):
        self.message = (
            f"Request body is larger than the limit of {MAX_REQUEST_BODY_BYTES} bytes."
        )
        super().__init__(status_code=413, detail=self.message)


class RequestBodyLimitMiddleware:
    """Count the body as the route reads it, so chunked uploads are limited too."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > MAX_REQUEST_BODY_BYTES:
                    raise RequestBodyTooLargeError()
            return message

        await self.app(scope, limited_receive, send)
