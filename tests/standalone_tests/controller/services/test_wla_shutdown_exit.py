"""Process exit while a landscape metadata RPC is still in flight.

``stop(join_timeout=...)`` only joins the landscape thread. The refresh runs
its metadata RPCs on a ``ThreadPoolExecutor``, and interpreter shutdown joins
those workers even though they are daemons. A call blocked for
``worker_tasks_timeout`` therefore keeps the process alive past the pod grace
period unless shutdown cancels the RPC first.
"""

import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import grpc

from exaflow.controller import DeploymentType
from exaflow.controller.services.controller_shutdown import begin_controller_shutdown
from exaflow.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    WorkerLandscapeAggregator,
)
from exaflow.controller.worker_client.app import WorkerClient
from exaflow.utils import AttrDict

REPO_ROOT = Path(__file__).resolve().parents[4]
_RPC_TIMEOUT_SECONDS = 30
_EXIT_TIMEOUT_SECONDS = 5


class _BlockingMetadataFuture:
    """A metadata RPC that returns only when cancelled or its deadline expires."""

    def __init__(self, timeout: float, ready: Path | None):
        self._timeout = timeout
        self._ready = ready
        self._cancelled = threading.Event()
        self._callbacks = []

    def result(self):
        if self._ready is not None:
            self._ready.write_text("blocked")
        if not self._cancelled.wait(self._timeout):
            raise grpc.RpcError()
        raise grpc.FutureCancelledError()

    def cancel(self):
        self._cancelled.set()
        for callback in list(self._callbacks):
            callback(self)
        return True

    def add_done_callback(self, callback):
        self._callbacks.append(callback)
        if self._cancelled.is_set():
            callback(self)


def _block_until_shutdown(ready: str, addresses: str) -> None:
    """Run one landscape refresh against an RPC that does not answer."""
    ready_path = Path(ready)
    Path(addresses).write_text('["127.0.0.1:1"]')

    def start_call(self, rpc_name, request, timeout):
        if rpc_name == "GetWorkerInfo":
            return _BlockingMetadataFuture(timeout, ready_path)
        finished = _BlockingMetadataFuture(0, None)
        finished.cancel()
        return finished

    WorkerClient.start_call = start_call
    aggregator = WorkerLandscapeAggregator(
        logger=logging.getLogger("wla-shutdown-exit"),
        update_interval=60,
        tasks_timeout=_RPC_TIMEOUT_SECONDS,
        deployment_type=DeploymentType.LOCAL,
        localworkers=AttrDict({"config_file": addresses}),
    )
    aggregator.start()
    deadline = time.monotonic() + 20
    while not ready_path.exists():
        if time.monotonic() > deadline:
            raise SystemExit("landscape metadata RPC did not start")
        time.sleep(0.05)
    # The executor thread is inside result(). Cancel, then leave the process
    # through normal interpreter shutdown.
    begin_controller_shutdown()
    aggregator.stop(join_timeout=1)
    raise SystemExit(0)


def test_process_exits_while_a_landscape_metadata_rpc_is_outstanding(tmp_path):
    """Shutdown must cancel the metadata RPC, not wait out worker_tasks_timeout."""
    ready = tmp_path / "rpc-blocked"
    addresses = tmp_path / "workers.json"
    output = tmp_path / "process.log"
    with output.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, __file__, str(ready), str(addresses)],
            cwd=REPO_ROOT,
            env={
                **os.environ,
                "PYTHONPATH": str(REPO_ROOT),
                "LOG_LEVEL": "INFO",
                "FRAMEWORK_LOG_LEVEL": "INFO",
            },
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    try:
        started = time.monotonic()
        while not ready.exists():
            if process.poll() is not None:
                raise AssertionError(output.read_text())
            if time.monotonic() - started > 60:
                raise AssertionError(output.read_text() or "RPC never started")
            time.sleep(0.05)
        deadline = time.monotonic() + _EXIT_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if process.poll() is not None:
                break
            time.sleep(0.05)
        assert process.poll() is not None, (
            "process still alive after shutdown with a landscape metadata RPC "
            "outstanding\n" + output.read_text()
        )
        assert process.returncode == 0, output.read_text()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


if __name__ == "__main__":
    _block_until_shutdown(sys.argv[1], sys.argv[2])
