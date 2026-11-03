"""Tests for the graceful controller shutdown in ``tasks.kill_controller``.

``kill_controller`` (the ``kill-controller`` invoke task) must stop Hypercorn
gracefully: it sends SIGTERM first so that shutdown hooks run, waits out the
controller shutdown budget (or ``--stop-timeout`` / ``CONTROLLER_STOP_TIMEOUT``)
polling every 0.5 s, and only SIGKILLs whatever is still alive afterwards.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
from invoke import Config
from invoke import Context

import tasks

# A real invoke Context so the @task-wrapped kill_controller accepts it.
_CTX = Context(Config())


class _Res:
    """Minimal stand-in for the ``CommandResult`` returned by ``run``."""

    def __init__(self, ok=True, stdout=""):
        self.ok = ok
        self.return_code = 0 if ok else 1
        self.stdout = stdout


def _run_recorder(alive_sequence):
    """Build a fake ``run`` recording commands and driving ``kill -0`` results.

    ``alive_sequence`` is an iterable of booleans describing, per ``kill -0``
    poll, whether the process is still alive. It is exhausted once it runs out
    (a process that keeps responding means "still alive").
    """
    calls = []
    it = iter(alive_sequence)

    def fake_run(c, cmd, **kwargs):
        calls.append(cmd)
        if cmd.startswith("kill -0"):
            return _Res(ok=next(it, True))
        return _Res(ok=True)

    return fake_run, calls


def _advance_clock():
    """Return a fake monotonic clock plus a sleep that advances it."""
    state = {"t": 0.0}

    def monotonic():
        return state["t"]

    def sleep(_seconds):
        state["t"] += _seconds

    return monotonic, sleep


def test_terminate_pids_sends_sigterm_first():
    """SIGTERM (``kill <pids>``) is issued before any SIGKILL."""
    fake_run, calls = _run_recorder(alive_sequence=[False])
    with (
        mock.patch.object(tasks, "run", side_effect=fake_run),
        mock.patch.object(tasks, "sleep", return_value=None),
        mock.patch("tasks.time.monotonic", return_value=0.0),
    ):
        tasks._terminate_pids(None, "4242")

    assert "kill 4242" in calls
    assert not any(cmd.startswith("kill -9") for cmd in calls)
    # SIGTERM precedes everything.
    assert calls.index("kill 4242") == 0


def test_terminate_pids_returns_when_process_exits():
    """No SIGKILL when the process acknowledges SIGTERM and exits."""
    fake_run, calls = _run_recorder(alive_sequence=[False])
    with (
        mock.patch.object(tasks, "run", side_effect=fake_run),
        mock.patch.object(tasks, "sleep", return_value=None),
        mock.patch("tasks.time.monotonic", return_value=0.0),
    ):
        tasks._terminate_pids(None, "4242")

    assert "kill 4242" in calls
    assert not any(cmd.startswith("kill -9") for cmd in calls)


def test_terminate_pids_sigkills_after_timeout():
    """A process ignoring SIGTERM is SIGKILLed once the grace period elapses."""
    fake_run, calls = _run_recorder(alive_sequence=[True])  # always still alive
    monotonic, sleep = _advance_clock()
    with (
        mock.patch.object(tasks, "run", side_effect=fake_run),
        mock.patch.object(tasks, "sleep", side_effect=sleep),
        mock.patch("tasks.time.monotonic", side_effect=monotonic),
    ):
        tasks._terminate_pids(None, "4242")

    assert "kill 4242" in calls, calls
    assert "kill -9 4242" in calls, calls
    # SIGTERM comes before SIGKILL.
    assert calls.index("kill 4242") < calls.index("kill -9 4242")


def test_terminate_pids_waits_for_the_hypercorn_drain():
    """The default wait covers the drain, RPC cancel, cleanup, and hooks.

    A fixed 10 s SIGKILL lands while Hypercorn is still inside graceful_timeout
    (T8). The wait is a maximum: a process that exits early returns immediately.
    """
    from exaflow.controller.quart import hypercorn_config as hc

    budget = (
        hc.graceful_timeout
        + hc.shutdown_rpc_cancel_timeout
        + hc.shutdown_cleanup_timeout
        + hc.shutdown_timeout
    )
    assert tasks.CONTROLLER_TERMINATE_TIMEOUT_SECONDS >= budget

    fake_run, calls = _run_recorder(alive_sequence=[True])
    monotonic, sleep = _advance_clock()
    with (
        mock.patch.object(tasks, "run", side_effect=fake_run),
        mock.patch.object(tasks, "sleep", side_effect=sleep),
        mock.patch("tasks.time.monotonic", side_effect=monotonic),
    ):
        tasks._terminate_pids(None, "4242")

    poll_count = sum(1 for cmd in calls if cmd.startswith("kill -0"))
    interval = tasks.CONTROLLER_TERMINATE_POLL_INTERVAL_SECONDS
    polls_in_the_wait = int(tasks.CONTROLLER_TERMINATE_TIMEOUT_SECONDS / interval)
    # One more kill -0 after the wait, immediately before SIGKILL.
    assert poll_count == polls_in_the_wait + 1


def test_kill_controller_stop_timeout_can_be_shorter_for_local_dev():
    """``--stop-timeout`` bounds the wait. The default stays long enough to drain."""
    lsof_out = (
        "COMMAND PID USER FD TYPE DEVICE SIZE/OFF NODE NAME\n"
        "hypercorn 222 user 5u IPv4 0x123 0t0 TCP *:5000 (LISTEN)\n"
    )
    calls = []
    sleeps = []
    state = {"t": 0.0}

    def fake_run(c, cmd, **kwargs):
        calls.append(cmd)
        if cmd.startswith("lsof"):
            return _Res(ok=True, stdout=lsof_out)
        if cmd.startswith("kill -0"):
            return _Res(ok=True)
        return _Res(ok=True)

    def sleep(seconds):
        sleeps.append(seconds)
        state["t"] += seconds

    with (
        mock.patch.object(tasks, "run", side_effect=fake_run),
        mock.patch.object(tasks, "sleep", side_effect=sleep),
        mock.patch("tasks.time.monotonic", side_effect=lambda: state["t"]),
        mock.patch.object(tasks, "get_controller_port", return_value=5000),
    ):
        tasks.kill_controller(_CTX, stop_timeout=1)

    assert "kill -9 222" in calls
    assert sum(sleeps) == 1
    assert sum(sleeps) < tasks.CONTROLLER_TERMINATE_TIMEOUT_SECONDS


def test_controller_stop_timeout_env_overrides_the_default(monkeypatch):
    lsof_out = (
        "COMMAND PID USER FD TYPE DEVICE SIZE/OFF NODE NAME\n"
        "hypercorn 222 user 5u IPv4 0x123 0t0 TCP *:5000 (LISTEN)\n"
    )
    sleeps = []
    state = {"t": 0.0}

    def fake_run(c, cmd, **kwargs):
        if cmd.startswith("lsof"):
            return _Res(ok=True, stdout=lsof_out)
        if cmd.startswith("kill -0"):
            return _Res(ok=True)
        return _Res(ok=True)

    def sleep(seconds):
        sleeps.append(seconds)
        state["t"] += seconds

    monkeypatch.setenv("CONTROLLER_STOP_TIMEOUT", "1")
    with (
        mock.patch.object(tasks, "run", side_effect=fake_run),
        mock.patch.object(tasks, "sleep", side_effect=sleep),
        mock.patch("tasks.time.monotonic", side_effect=lambda: state["t"]),
        mock.patch.object(tasks, "get_controller_port", return_value=5000),
    ):
        tasks.kill_controller(_CTX)

    assert sum(sleeps) == 1


def test_terminate_pids_rechecks_before_sigkill():
    """A pid that exits during the last sleep must not be SIGKILLed.

    The in-loop poll can still see the process, and it can exit before
    ``kill -9``. SIGKILL only the pids that are alive at that moment (T8).
    """
    polls_in_loop = int(
        tasks.CONTROLLER_TERMINATE_TIMEOUT_SECONDS
        / tasks.CONTROLLER_TERMINATE_POLL_INTERVAL_SECONDS
    )
    state = {"checks": 0}
    calls = []

    def fake_run(c, cmd, **kwargs):
        calls.append(cmd)
        if cmd.startswith("kill -0"):
            state["checks"] += 1
            # Alive for every poll inside the wait, gone on the re-check.
            return _Res(ok=state["checks"] <= polls_in_loop)
        return _Res(ok=True)

    monotonic, sleep = _advance_clock()
    with (
        mock.patch.object(tasks, "run", side_effect=fake_run),
        mock.patch.object(tasks, "sleep", side_effect=sleep),
        mock.patch("tasks.time.monotonic", side_effect=monotonic),
    ):
        tasks._terminate_pids(None, "4242")

    assert not any(cmd.startswith("kill -9") for cmd in calls), calls
    assert state["checks"] == polls_in_loop + 1


def test_terminate_pids_multiple_pids_sigkill_only_alive():
    """Only the still-alive pids receive SIGKILL after the grace period."""
    state = {"t": 0.0}
    calls = []

    def fake_run(c, cmd, **kwargs):
        calls.append(cmd)
        if cmd.startswith("kill -0"):
            pid = cmd.split()[-1]
            return _Res(
                ok={
                    "1": False,  # exits immediately
                    "2": True,  # ignores SIGTERM
                }[pid]
            )
        return _Res(ok=True)

    def monotonic():
        return state["t"]

    def sleep(_seconds):
        state["t"] += _seconds

    with (
        mock.patch.object(tasks, "run", side_effect=fake_run),
        mock.patch.object(tasks, "sleep", side_effect=sleep),
        mock.patch("tasks.time.monotonic", side_effect=monotonic),
    ):
        tasks._terminate_pids(None, "1 2")

    assert "kill 1 2" in calls
    # pid 1 exited, so only pid 2 is SIGKILLed.
    assert "kill -9 2" in calls
    assert "kill -9 1" not in calls


def test_kill_controller_sends_sigterm_before_sigkill():
    """kill-controller SIGTERMes the Hypercorn pid found via lsof, then SIGKILLs."""
    lsof_out = (
        "COMMAND PID USER FD TYPE DEVICE SIZE/OFF NODE NAME\n"
        "hypercorn 222 user 5u IPv4 0x123 0t0 TCP *:5000 (LISTEN)\n"
    )
    calls = []
    state = {"t": 0.0}

    def fake_run(c, cmd, **kwargs):
        calls.append(cmd)
        if cmd.startswith("lsof"):
            return _Res(ok=True, stdout=lsof_out)
        if cmd.startswith("kill -0"):
            return _Res(ok=True)  # stays alive -> forces SIGKILL
        return _Res(ok=True)

    def monotonic():
        return state["t"]

    def sleep(_seconds):
        state["t"] += _seconds

    with (
        mock.patch.object(tasks, "run", side_effect=fake_run),
        mock.patch.object(tasks, "sleep", side_effect=sleep),
        mock.patch("tasks.time.monotonic", side_effect=monotonic),
        mock.patch.object(tasks, "get_controller_port", return_value=5000),
    ):
        tasks.kill_controller(_CTX)

    assert "kill 222" in calls
    assert "kill -9 222" in calls
    assert calls.index("kill 222") < calls.index("kill -9 222")


def test_kill_controller_returns_when_controller_exits():
    """kill-controller returns without SIGKILL once the controller exits."""
    lsof_out = (
        "COMMAND PID USER FD TYPE DEVICE SIZE/OFF NODE NAME\n"
        "python 333 user 5u IPv4 0x123 0t0 TCP *:5000 (LISTEN)\n"
    )
    calls = []

    def fake_run(c, cmd, **kwargs):
        calls.append(cmd)
        if cmd.startswith("lsof"):
            return _Res(ok=True, stdout=lsof_out)
        if cmd.startswith("kill -0"):
            return _Res(ok=False)  # gone right after SIGTERM
        return _Res(ok=True)

    with (
        mock.patch.object(tasks, "run", side_effect=fake_run),
        mock.patch.object(tasks, "sleep", return_value=None),
        mock.patch("tasks.time.monotonic", return_value=0.0),
        mock.patch.object(tasks, "get_controller_port", return_value=5000),
    ):
        tasks.kill_controller(_CTX)

    assert "kill 333" in calls
    assert not any(cmd.startswith("kill -9") for cmd in calls)


def test_kill_controller_uses_stopping_message():
    """The message no longer says 'Killing previous Hypercorn instances...'."""
    lsof_out = (
        "COMMAND PID USER FD TYPE DEVICE SIZE/OFF NODE NAME\n"
        "hypercorn 222 user 5u IPv4 0x123 0t0 TCP *:5000 (LISTEN)\n"
    )
    messages = []

    def fake_run(c, cmd, **kwargs):
        if cmd.startswith("lsof"):
            return _Res(ok=True, stdout=lsof_out)
        if cmd.startswith("kill -0"):
            return _Res(ok=False)
        return _Res(ok=True)

    def fake_message(msg, level=None):
        messages.append(msg)

    with (
        mock.patch.object(tasks, "run", side_effect=fake_run),
        mock.patch.object(tasks, "sleep", return_value=None),
        mock.patch("tasks.time.monotonic", return_value=0.0),
        mock.patch.object(tasks, "get_controller_port", return_value=5000),
        mock.patch.object(tasks, "message", side_effect=fake_message),
    ):
        tasks.kill_controller(_CTX)

    assert any("Stopping previous Hypercorn instances..." in m for m in messages)
    assert not any("Killing previous Hypercorn instances..." in m for m in messages)


@pytest.mark.parametrize("command", ["python3.11", "python3.1"])
def test_kill_controller_recognizes_versioned_python(command):
    """lsof shows at most 9 command characters unless +c 0 is set.

    python3.11 is therefore python3.1, which is neither "python" nor
    "hypercorn", and the controller must still be stopped.
    """
    lsof_out = (
        "COMMAND PID USER FD TYPE DEVICE SIZE/OFF NODE NAME\n"
        f"{command} 333 user 5u IPv4 0x123 0t0 TCP *:5000 (LISTEN)\n"
    )
    calls = []
    messages = []

    def fake_run(c, cmd, **kwargs):
        calls.append(cmd)
        if cmd.startswith("lsof"):
            return _Res(ok=True, stdout=lsof_out)
        if cmd.startswith("kill -0"):
            return _Res(ok=False)
        return _Res(ok=False, stdout="")

    def fake_message(msg, level=None):
        messages.append(msg)

    with (
        mock.patch.object(tasks, "run", side_effect=fake_run),
        mock.patch.object(tasks, "sleep", return_value=None),
        mock.patch("tasks.time.monotonic", return_value=0.0),
        mock.patch.object(tasks, "get_controller_port", return_value=5000),
        mock.patch.object(tasks, "message", side_effect=fake_message),
    ):
        tasks.kill_controller(_CTX)

    assert any(cmd.startswith("lsof") and "+c 0" in cmd for cmd in calls)
    assert "kill 333" in calls
    assert not any("is used by" in msg for msg in messages)


def test_terminate_poll_is_not_stretched_by_the_spinner():
    """The 0.5 s poll must not also pay for the invoke spinner's sleep."""
    sleeps = []
    state = {"t": 0.0}

    class _Promise:
        def __init__(self, result):
            self.runner = SimpleNamespace(process_is_finished=True, stderr=None)
            self._result = result

        def join(self):
            return self._result

    def fake_c_run(cmd, **kwargs):
        result = _Res(ok=cmd.startswith("kill -0"))
        if kwargs.get("asynchronous"):
            return _Promise(result)
        return result

    def sleep(seconds):
        sleeps.append(seconds)
        state["t"] += seconds

    with (
        mock.patch.object(tasks, "sleep", side_effect=sleep),
        mock.patch("tasks.time.monotonic", side_effect=lambda: state["t"]),
    ):
        tasks._terminate_pids(SimpleNamespace(run=fake_c_run), "4242")

    interval = tasks.CONTROLLER_TERMINATE_POLL_INTERVAL_SECONDS
    polls = int(tasks.CONTROLLER_TERMINATE_TIMEOUT_SECONDS / interval)
    assert sleeps == [interval] * polls
