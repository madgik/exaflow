"""Container PID 1 ignores SIGTERM unless the worker installs a handler.

The Dockerfile exec form delivers SIGTERM to the gRPC server. The kernel
drops that signal for PID 1 when the process has no handler, so the server
has to call server.stop and leave wait_for_termination.
"""

import os
import signal
import subprocess
import sys
import threading

import pytest

_READY = "Server starting on"
_CHILD = (
    "import signal\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "from exaflow.worker.grpc_server import serve\n"
    "serve()\n"
)


def test_worker_process_exits_on_sigterm(tmp_path):
    env = os.environ.copy()
    env.pop("EXAFLOW_WORKER_CONFIG_FILE", None)
    env.update(
        {
            "GRPC_IP": "127.0.0.1",
            "GRPC_PORT": "0",
            "GRPC_MAX_WORKERS": "2",
            "DATA_PATH": str(tmp_path),
            "DUCKDB_PATH": str(tmp_path / "data_models.duckdb"),
            "WORKER_IDENTIFIER": "sigterm-test",
            "WORKER_ROLE": "localworker",
            "FEDERATION": "test",
            "LOG_LEVEL": "INFO",
            "FRAMEWORK_LOG_LEVEL": "INFO",
        }
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", _CHILD],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    ready = threading.Event()
    output: list[str] = []

    def _read() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            output.append(line)
            if _READY in line:
                ready.set()

    reader = threading.Thread(target=_read, daemon=True)
    reader.start()
    try:
        assert ready.wait(30), "worker did not start:\n" + "".join(output)
        proc.send_signal(signal.SIGTERM)
        try:
            code = proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
            pytest.fail("worker stayed up after SIGTERM:\n" + "".join(output))
        assert code == 0, "".join(output)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
