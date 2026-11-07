"""App served by a real Uvicorn process in ``test_uvicorn_shutdown.py``.

It wires the controller's own shutdown pieces (``run_blocking``, the in-flight
RPC registry and the ``endpoints.shutdown`` lifespan hook) around a gRPC call
that never finishes by itself, and appends what happens to the file named by
``SHUTDOWN_PROBE_EVENTS``, one ``<unix time> <event>`` line each.
"""

import os
import threading
import time
from concurrent import futures
from contextlib import asynccontextmanager

import grpc
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from exaflow.controller.http import endpoints
from exaflow.controller.services import set_worker_landscape_aggregator
from exaflow.controller.services.analysis_execution import run_blocking
from exaflow.controller.services.analysis_execution import start_analysis_execution
from exaflow.controller.services.controller_shutdown import register_inflight_rpc


def _event(name: str) -> None:
    with open(os.environ["SHUTDOWN_PROBE_EVENTS"], "a") as events:
        events.write(f"{time.time():.3f} {name}\n")


class _Landscape:
    def stop(self, join_timeout=None):
        _event("landscape-stopped")


def _never_finishes(request, context):
    cancelled = threading.Event()
    context.add_callback(cancelled.set)
    cancelled.wait(120)
    return b""


_server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
_server.add_generic_rpc_handlers(
    (
        grpc.method_handlers_generic_handler(
            "probe.Probe",
            {"Wait": grpc.unary_unary_rpc_method_handler(_never_finishes)},
        ),
    )
)
_port = _server.add_insecure_port("127.0.0.1:0")
_server.start()
_wait = grpc.insecure_channel(f"127.0.0.1:{_port}").unary_unary("/probe.Probe/Wait")


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    set_worker_landscape_aggregator(_Landscape())
    start_analysis_execution()
    try:
        yield
    finally:
        await endpoints.shutdown()


app = FastAPI(lifespan=_lifespan)


def _analysis(name: str) -> str:
    call = _wait.future(b"", timeout=100)
    register_inflight_rpc(call)
    _event(f"{name}:rpc-started")
    try:
        call.result()
    except grpc.FutureCancelledError:
        _event(f"{name}:rpc-cancelled")
        raise
    return "done"


@app.get("/ping", response_class=PlainTextResponse)
async def ping():
    return "pong"


@app.get("/analysis/{name}", response_class=PlainTextResponse)
async def analysis(name: str):
    try:
        return await run_blocking(_analysis, name)
    finally:
        # Like the aggregation cleanup in the strategy: it runs after the
        # analysis thread ended, also when shutdown ended it.
        await run_blocking(_event, f"{name}:cleanup-ran")
