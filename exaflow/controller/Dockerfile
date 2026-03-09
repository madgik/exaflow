FROM python:3.10.18-slim-bullseye AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=2.3.2 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    CODE_PATH="/opt/code"

WORKDIR $CODE_PATH

COPY pyproject.toml poetry.lock ./
RUN pip install --no-cache-dir --prefix "$POETRY_HOME" poetry==$POETRY_VERSION \
    && PYTHONPATH="/opt/poetry/lib/python3.10/site-packages" python -m poetry install --only main --only controller --no-root \
    && rm -rf /root/.cache/pip /root/.cache/pypoetry "$POETRY_HOME"

FROM python:3.10.18-slim-bullseye
LABEL maintainer="Thanasis Karampatsis <tkarabatsis@athenarc.gr>"

#######################################################
# Timezone
#######################################################
ENV TZ=Etc/GMT
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

#######################################################
# Env & workdir
#######################################################
ENV PYTHONUNBUFFERED=1 \
    CODE_PATH="/opt/code"
WORKDIR $CODE_PATH

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

#######################################################
# Copying the codebase
#######################################################
COPY exaflow/ ./exaflow/
RUN rm -rf exaflow/worker

RUN groupadd -g 10001 appuser \
    && useradd -m -u 10001 -g appuser -s /bin/bash appuser \
    && chown -R appuser:appuser /opt/code

#######################################################
# Define Hypercorn needed env variables and start hypercorn
#######################################################
ENV PYTHONPATH=$CODE_PATH
USER appuser
CMD ["sh", "-c", "python -m hypercorn --config python:exaflow.controller.quart.hypercorn_config -b 0.0.0.0:5000 exaflow/controller/quart/app:app"]
