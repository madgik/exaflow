FROM python:3.10.18-slim-bullseye AS builder

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

COPY --from=ghcr.io/astral-sh/uv:0.11.0 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --only-group aggregation-server

FROM python:3.10.18-slim-bullseye
LABEL maintainer="Konstantinos Filippopolitis <kfilippopolitis@athenarc.gr>"

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"
COPY --from=builder /app/.venv /app/.venv

COPY exaflow/aggregation_server ./exaflow/aggregation_server
COPY exaflow/aggregation_serialization.py ./exaflow/aggregation_serialization.py
COPY exaflow/protos ./exaflow/protos

EXPOSE 50051
ENV AGG_SERVER_CONFIG_FILE=/app/exaflow/aggregation_server/config.toml

ENTRYPOINT ["python", "-m", "exaflow.aggregation_server.server"]
