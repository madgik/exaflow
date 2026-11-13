"""Shutdown of the controller under a real Uvicorn process.

The unit tests of the shutdown helpers pass with any assumption about how
Uvicorn and FastAPI deliver cancellation. These start ``python -m uvicorn``
with the controller's settings (only the drain is shortened) and send it
SIGTERM while analyses are in flight.
"""

import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from exaflow.controller.http import server_config

REPO_ROOT = Path(__file__).resolve().parents[3]
GRACEFUL_TIMEOUT = 3
STARTUP_TIMEOUT = 90
# Uvicorn re-raises the SIGTERM it handled once shutdown is complete.
CLEAN_EXIT = (0, -signal.SIGTERM)


class _Probe:
    """One Uvicorn process serving ``shutdown_probe_app``."""

    def __init__(self, tmp_path):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.events_file = tmp_path / "events.txt"
        self.output_file = tmp_path / "uvicorn.log"
        self._connections = []
        env = dict(
            os.environ,
            PYTHONPATH=str(REPO_ROOT),
            SHUTDOWN_PROBE_EVENTS=str(self.events_file),
            LOG_LEVEL="INFO",
            FRAMEWORK_LOG_LEVEL="INFO",
        )
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "tests.standalone_tests.controller.shutdown_probe_app:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--workers",
                str(server_config.workers),
                "--timeout-graceful-shutdown",
                str(GRACEFUL_TIMEOUT),
            ],
            cwd=REPO_ROOT,
            env=env,
            stdout=self.output_file.open("w"),
            stderr=subprocess.STDOUT,
        )
        self._wait_until_serving()

    def _wait_until_serving(self):
        deadline = time.monotonic() + STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            assert self.process.poll() is None, self.output()
            try:
                url = f"http://127.0.0.1:{self.port}/ping"
                urllib.request.urlopen(url, timeout=1).read()
                return
            except OSError:
                time.sleep(0.2)
        raise AssertionError(f"server did not start\n{self.output()}")

    def start_analysis(self, name):
        """Open a request that stays in flight, and wait until its RPC runs."""
        connection = socket.create_connection(("127.0.0.1", self.port))
        connection.sendall(f"GET /analysis/{name} HTTP/1.1\r\nHost: x\r\n\r\n".encode())
        self._connections.append(connection)
        deadline = time.monotonic() + 30
        while f"{name}:rpc-started" not in self.events():
            assert time.monotonic() < deadline, self.output()
            time.sleep(0.05)
        return connection

    def events(self):
        if not self.events_file.exists():
            return {}
        lines = (line.split() for line in self.events_file.read_text().splitlines())
        return {name: float(stamp) for stamp, name in lines}

    def output(self):
        return self.output_file.read_text()

    def terminate(self):
        """Send SIGTERM; return when it was sent."""
        sent_at = time.time()
        self.process.send_signal(signal.SIGTERM)
        return sent_at

    def exit_code(self):
        timeout = GRACEFUL_TIMEOUT + 30
        try:
            return self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pytest.fail(f"still running {timeout} s after SIGTERM\n{self.output()}")

    def close(self):
        for connection in self._connections:
            connection.close()
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()


@pytest.fixture
def probe(tmp_path):
    started = _Probe(tmp_path)
    try:
        yield started
    finally:
        started.close()


def test_analysis_running_at_drain_expiry_is_cancelled_and_the_process_exits_cleanly(
    probe,
):
    """Uvicorn cancels the request task at drain expiry, but the analysis thread
    keeps running. The lifespan hook has to cancel its RPC, or the process waits
    for the RPC deadline."""
    probe.start_analysis("a")

    sent_at = probe.terminate()
    returncode = probe.exit_code()

    events = probe.events()
    assert returncode in CLEAN_EXIT, probe.output()
    assert "still running" not in probe.output()
    assert events["a:rpc-cancelled"] - sent_at >= GRACEFUL_TIMEOUT - 1
    assert events["a:rpc-cancelled"] - sent_at < GRACEFUL_TIMEOUT + 10
    assert "landscape-stopped" in events
    # The cleanup of the cancelled analysis still runs before the process exits.
    assert events["a:cleanup-ran"] >= events["a:rpc-cancelled"]


def test_client_disconnect_during_the_drain_does_not_cancel_other_analyses(probe):
    """A disconnect is not a shutdown: analyses that are draining keep running
    until the drain is over."""
    probe.start_analysis("a")
    gone = probe.start_analysis("b")

    sent_at = probe.terminate()
    time.sleep(0.5)
    gone.close()
    returncode = probe.exit_code()

    events = probe.events()
    assert returncode in CLEAN_EXIT, probe.output()
    assert events["a:rpc-cancelled"] - sent_at >= GRACEFUL_TIMEOUT - 1


def test_queued_analysis_is_drained_cancelled_and_left_unacked(probe):
    request = urllib.request.Request(
        f"http://127.0.0.1:{probe.port}/queue/queued", data=b"", method="POST"
    )
    urllib.request.urlopen(request, timeout=5).read()
    deadline = time.monotonic() + 10
    while "queued:rpc-started" not in probe.events():
        assert time.monotonic() < deadline, probe.output()
        time.sleep(0.05)
    probe.terminate()
    assert probe.exit_code() in CLEAN_EXIT, probe.output()
    events = probe.events()
    assert events["queue-drained"] <= events["queued:rpc-cancelled"]
    assert events["queued:cleanup-ran"] >= events["queued:rpc-cancelled"]
    assert "queued:acked" not in events
