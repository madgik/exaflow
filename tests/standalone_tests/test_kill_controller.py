"""Tests for the graceful controller shutdown in ``tasks.kill_controller``.

``kill_controller`` (the ``kill-controller`` invoke task) must stop Hypercorn
gracefully: it sends SIGTERM first so that shutdown hooks run, waits up to 10 s
polling every 0.5 s, and only SIGKILLs whatever is still alive afterwards.
"""

from unittest import mock

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


def test_terminate_pids_waits_full_grace_period():
    """It polls for the full 10 s (0.5 s interval => 20 polls) before SIGKILL."""
    fake_run, calls = _run_recorder(alive_sequence=[True])
    monotonic, sleep = _advance_clock()
    with (
        mock.patch.object(tasks, "run", side_effect=fake_run),
        mock.patch.object(tasks, "sleep", side_effect=sleep),
        mock.patch("tasks.time.monotonic", side_effect=monotonic),
    ):
        tasks._terminate_pids(None, "4242")

    poll_count = sum(1 for cmd in calls if cmd.startswith("kill -0"))
    # 10 s / 0.5 s = 20 polls before the deadline stops the loop.
    assert poll_count == 20


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
