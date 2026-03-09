FROM python:3.10.18-slim-bullseye AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=2.3.2 \
    POETRY_NO_INTERACTION=1 \
    POETRY_HOME="/opt/poetry"

WORKDIR /app

RUN pip install --no-cache-dir --prefix "$POETRY_HOME" poetry==$POETRY_VERSION \
    && PYTHONPATH="/opt/poetry/lib/python3.10/site-packages" python -m poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock ./
RUN PYTHONPATH="/opt/poetry/lib/python3.10/site-packages" python -m poetry install --only aggregation-server --no-root --no-ansi \
    && rm -rf /root/.cache/pip /root/.cache/pypoetry "$POETRY_HOME"

FROM python:3.10.18-slim-bullseye
LABEL maintainer="Konstantinos Filippopolitis <kfilippopolitis@athenarc.gr>"

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

COPY exaflow/aggregation_server ./exaflow/aggregation_server
COPY exaflow/protos ./exaflow/protos

EXPOSE 50051
ENV AGG_SERVER_CONFIG_FILE=/app/exaflow/aggregation_server/config.toml

ENTRYPOINT ["python", "-m", "exaflow.aggregation_server.server"]
