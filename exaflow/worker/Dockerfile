FROM python:3.10.18-slim-bullseye AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    CODE_PATH="/opt/code"

COPY --from=ghcr.io/astral-sh/uv:0.11.0 /uv /uvx /bin/

WORKDIR $CODE_PATH

COPY pyproject.toml uv.lock ./
# To restore Flower deps, add `--group flower` to the uv sync (see MIP-1203).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-default-groups --group worker

FROM python:3.10.18-slim-bullseye
LABEL maintainer="Thanasis Karampatsis <tkarabatsis@athenarc.gr>"

#######################################################
# Setting up timezone
#######################################################
ENV TZ=Etc/GMT
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

#######################################################
# Setting up env variables and workdir
#######################################################
ENV PYTHONUNBUFFERED=1 \
    DATA_PATH="/opt/data" \
    CODE_PATH="/opt/code" \
    PATH="/opt/code/.venv/bin:$PATH"
WORKDIR $CODE_PATH

ENV PYTHONPATH=$CODE_PATH
RUN mkdir $DATA_PATH
VOLUME $DATA_PATH

COPY --from=builder $CODE_PATH/.venv $CODE_PATH/.venv

#######################################################
# Copying the codebase
#######################################################
COPY exaflow/ ./exaflow/
RUN rm -rf exaflow/controller

RUN groupadd -g 10001 appuser \
    && useradd -m -u 10001 -g appuser -s /bin/bash appuser \
    && chown -R appuser:appuser /opt/code /opt/data /home/appuser

#######################################################
# Setup bootstrap file
#######################################################
COPY exaflow/worker/bootstrap.sh /home/bootstrap.sh
RUN chmod 775 /home/bootstrap.sh

USER appuser
CMD ["sh", "-c", "/home/bootstrap.sh"]
