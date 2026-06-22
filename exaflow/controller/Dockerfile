FROM python:3.10.18-slim-bullseye AS builder

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    CODE_PATH="/opt/code"

COPY --from=ghcr.io/astral-sh/uv:0.11.0 /uv /uvx /bin/

WORKDIR $CODE_PATH

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-default-groups --group controller

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
    PYTHONDONTWRITEBYTECODE=1 \
    CODE_PATH="/opt/code" \
    PATH="/opt/code/.venv/bin:$PATH"
WORKDIR $CODE_PATH

RUN groupadd -g 10001 appuser \
    && useradd -m -u 10001 -g appuser -s /bin/bash appuser

COPY --chown=10001:10001 --from=builder $CODE_PATH/.venv $CODE_PATH/.venv

#######################################################
# Copying the codebase
#######################################################
COPY --chown=10001:10001 exaflow/ ./exaflow/
RUN rm -rf exaflow/worker

#######################################################
# Define Hypercorn needed env variables and start hypercorn
#######################################################
ENV PYTHONPATH=$CODE_PATH
USER appuser
CMD ["sh", "-c", "python -m hypercorn --config python:exaflow.controller.quart.hypercorn_config -b 0.0.0.0:5000 exaflow/controller/quart/app:app"]
