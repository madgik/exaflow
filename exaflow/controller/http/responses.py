from starlette.responses import PlainTextResponse

# Starlette appends "; charset=utf-8" to text/* media types. The served
# header is then "text/html; charset=utf-8", which is the Quart contract.
HTML_MEDIA_TYPE = "text/html"


def html_response(body: str, status_code: int = 200) -> PlainTextResponse:
    return PlainTextResponse(body, status_code=status_code, media_type=HTML_MEDIA_TYPE)
