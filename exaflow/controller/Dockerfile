FROM python:3.11.18-slim-trixie AS builder

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    CODE_PATH="/opt/code"

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /uvx /bin/

WORKDIR $CODE_PATH

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-default-groups --group controller

FROM python:3.11.18-slim-trixie
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
# Uvicorn in-process. workers 1 is the server itself (a higher value is a
# supervisor at PID 1). 120 matches server_config.graceful_timeout.
#######################################################
ENV PYTHONPATH=$CODE_PATH
USER appuser
CMD ["python", "-m", "uvicorn", "exaflow.controller.http.app:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "1", "--timeout-graceful-shutdown", "120", "--log-config", "exaflow/controller/http/log_config.json"]
