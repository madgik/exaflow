"""Tests for the graceful controller shutdown in ``tasks.kill_controller``.

``kill_controller`` (the ``kill-controller`` invoke task) must stop Hypercorn
gracefully: it sends SIGTERM first so that shutdown hooks run, waits out the
controller shutdown budget (or ``--stop-timeout`` / ``CONTROLLER_STOP_TIMEOUT``)
polling every 0.5 s, and only SIGKILLs whatever is still alive afterwards.
"""

import os
from contextlib import ExitStack
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from invoke import Config
from invoke import Context

import tasks
from exaflow.controller.services import controller_shutdown as budget

# A real invoke Context so the @task-wrapped kill_controller accepts it.
_CTX = Context(Config())

_LSOF_HYPERCORN = (
    "COMMAND PID USER FD TYPE DEVICE SIZE/OFF NODE NAME\n"
    "hypercorn 222 user 5u IPv4 0x123 0t0 TCP *:5000 (LISTEN)\n"
)


class _Res:
    """Minimal stand-in for the ``CommandResult`` returned by ``run``."""

    def __init__(self, ok=True, stdout=""):
        self.ok = ok
        self.return_code = 0 if ok else 1
        self.stdout = stdout


def _run_recorder(lsof_out=""):
    """Build a fake ``run`` recording the commands it is given."""
    calls = []

    def fake_run(c, cmd, **kwargs):
        calls.append(cmd)
        if cmd.startswith("lsof"):
            return _Res(ok=True, stdout=lsof_out)
        return _Res(ok=True)

    return fake_run, calls


def _liveness(alive_sequence):
    """Fake ``tasks._pid_alive`` driven by a sequence of booleans.

    One value per poll. Once the sequence runs out, a process that kept
    responding is still alive.
    """
    it = iter(alive_sequence)
    polls = []

    def pid_alive(pid):
        polls.append(pid)
        return next(it, True)

    return pid_alive, polls


def _advance_clock():
    """Return a fake monotonic clock plus a sleep that advances it."""
    state = {"t": 0.0}

    def monotonic():
        return state["t"]

    def sleep(_seconds):
        state["t"] += _seconds

    return monotonic, sleep


@contextmanager
def _patched(fake_run, pid_alive, monotonic, sleep, **extra):
    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(tasks, "run", side_effect=fake_run))
        stack.enter_context(
            mock.patch.object(tasks, "_pid_alive", side_effect=pid_alive)
        )
        stack.enter_context(mock.patch.object(tasks, "sleep", side_effect=sleep))
        stack.enter_context(mock.patch("tasks.time.monotonic", side_effect=monotonic))
        for name, value in extra.items():
            stack.enter_context(mock.patch.object(tasks, name, value))
        yield


def test_terminate_pids_sends_sigterm_first():
    """SIGTERM (``kill <pids>``) is issued before any SIGKILL."""
    fake_run, calls = _run_recorder()
    pid_alive, _ = _liveness([False])
    monotonic, sleep = _advance_clock()
    with _patched(fake_run, pid_alive, monotonic, sleep):
        tasks._terminate_pids(None, "4242")

    assert "kill 4242" in calls
    assert not any(cmd.startswith("kill -9") for cmd in calls)
    assert calls.index("kill 4242") == 0


def test_terminate_pids_returns_when_process_exits():
    """No SIGKILL when the process acknowledges SIGTERM and exits."""
    fake_run, calls = _run_recorder()
    pid_alive, polls = _liveness([True, True, False])
    monotonic, sleep = _advance_clock()
    with _patched(fake_run, pid_alive, monotonic, sleep):
        tasks._terminate_pids(None, "4242")

    assert "kill 4242" in calls
    assert not any(cmd.startswith("kill -9") for cmd in calls)
    assert len(polls) == 3


def test_terminate_pids_sigkills_after_timeout():
    """A process ignoring SIGTERM is SIGKILLed once the grace period elapses."""
    fake_run, calls = _run_recorder()
    pid_alive, _ = _liveness([])  # always still alive
    monotonic, sleep = _advance_clock()
    with _patched(fake_run, pid_alive, monotonic, sleep):
        tasks._terminate_pids(None, "4242")

    assert "kill 4242" in calls, calls
    assert "kill -9 4242" in calls, calls
    assert calls.index("kill 4242") < calls.index("kill -9 4242")


def test_default_wait_equals_the_hypercorn_shutdown_budget():
    """tasks.py cannot import exaflow (it takes seconds), so it repeats the sum."""
    assert tasks.CONTROLLER_TERMINATE_TIMEOUT_SECONDS == (
        budget.GRACEFUL_TIMEOUT
        + budget.SHUTDOWN_RPC_CANCEL_TIMEOUT
        + budget.SHUTDOWN_CLEANUP_TIMEOUT
        + budget.SHUTDOWN_TIMEOUT
    )


def test_tasks_module_does_not_import_the_exaflow_runtime():
    """Importing exaflow runs algorithm discovery, 3+ s on every ``inv`` call."""
    source = open(tasks.__file__).read()

    assert "from exaflow" not in source
    assert "import exaflow" not in source


def test_shared_tmp_directories_tolerate_a_concurrent_creator(tmp_path, monkeypatch):
    """xdist workers import tasks together and both create /tmp/exaflow.

    A check-then-mkdir raises FileExistsError when the other worker creates
    the directory first, and the workers then collect different tests.
    """
    target = tmp_path / "exaflow"
    real_mkdir = os.mkdir

    def peer_creates_it_first(path, mode=0o777, *, dir_fd=None):
        if Path(path) == target and not target.exists():
            real_mkdir(path, mode)
        if dir_fd is None:
            return real_mkdir(path, mode)
        return real_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "mkdir", peer_creates_it_first)

    tasks._ensure_directory(target)

    assert target.is_dir()
    source = Path(tasks.__file__).read_text()
    assert "if not OUTDIR.exists()" not in source
    assert "if not CLEANUP_DIR.exists()" not in source


def test_terminate_pids_waits_for_the_hypercorn_drain():
    """The default wait covers the drain, RPC cancel, cleanup, and hooks.

    A fixed 10 s SIGKILL lands while Hypercorn is still inside graceful_timeout
    (T8). The wait is a maximum: a process that exits early returns immediately.
    """
    fake_run, calls = _run_recorder()
    pid_alive, polls = _liveness([])  # always still alive
    monotonic, sleep = _advance_clock()
    with _patched(fake_run, pid_alive, monotonic, sleep):
        tasks._terminate_pids(None, "4242")

    interval = tasks.CONTROLLER_TERMINATE_POLL_INTERVAL_SECONDS
    polls_in_the_wait = int(tasks.CONTROLLER_TERMINATE_TIMEOUT_SECONDS / interval)
    # One more poll after the wait, immediately before SIGKILL.
    assert len(polls) == polls_in_the_wait + 1


def test_pid_alive_reports_a_running_and_a_gone_process():
    assert tasks._pid_alive(str(os.getpid()))
    with mock.patch.object(tasks.os, "kill", side_effect=ProcessLookupError):
        assert not tasks._pid_alive("999999")
    with mock.patch.object(tasks.os, "kill", side_effect=PermissionError):
        assert tasks._pid_alive("1")


def test_kill_controller_stop_timeout_can_be_shorter_for_local_dev():
    """``--stop-timeout`` bounds the wait. The default stays long enough to drain."""
    fake_run, calls = _run_recorder(_LSOF_HYPERCORN)
    pid_alive, _ = _liveness([])
    sleeps = []
    monotonic, advance = _advance_clock()

    def sleep(seconds):
        sleeps.append(seconds)
        advance(seconds)

    with _patched(
        fake_run,
        pid_alive,
        monotonic,
        sleep,
        get_controller_port=mock.Mock(return_value=5000),
    ):
        tasks.kill_controller(_CTX, stop_timeout=1)

    assert "kill -9 222" in calls
    assert sum(sleeps) == 1
    assert sum(sleeps) < tasks.CONTROLLER_TERMINATE_TIMEOUT_SECONDS


def test_controller_stop_timeout_env_overrides_the_default(monkeypatch):
    fake_run, calls = _run_recorder(_LSOF_HYPERCORN)
    pid_alive, _ = _liveness([])
    sleeps = []
    monotonic, advance = _advance_clock()

    def sleep(seconds):
        sleeps.append(seconds)
        advance(seconds)

    monkeypatch.setenv("CONTROLLER_STOP_TIMEOUT", "1")
    with _patched(
        fake_run,
        pid_alive,
        monotonic,
        sleep,
        get_controller_port=mock.Mock(return_value=5000),
    ):
        tasks.kill_controller(_CTX)

    assert sum(sleeps) == 1


class _Stop(Exception):
    """Ends a task right after it stopped the controller."""


def _restart_timeout_used_by(task, **kwargs):
    seen = []

    def fake_kill(c, timeout):
        seen.append(timeout)
        raise _Stop

    with mock.patch.object(tasks, "_kill_controller", side_effect=fake_kill):
        with mock.patch.object(tasks, "validate_algorithm_folders", return_value=None):
            with pytest.raises(_Stop):
                task(_CTX, **kwargs)
    return seen


@pytest.mark.parametrize("task_name", ["start_controller", "cleanup"])
def test_implicit_controller_stops_do_not_wait_out_the_drain(task_name, monkeypatch):
    """A restart or cleanup stops the controller as a side effect: waiting 195 s
    for a running analysis to drain would look like a hang."""
    monkeypatch.delenv("CONTROLLER_STOP_TIMEOUT", raising=False)

    seen = _restart_timeout_used_by(getattr(tasks, task_name))

    assert seen == [tasks.CONTROLLER_RESTART_TIMEOUT_SECONDS]
    assert seen[0] < tasks.CONTROLLER_TERMINATE_TIMEOUT_SECONDS


@pytest.mark.parametrize("task_name", ["start_controller", "cleanup"])
def test_controller_stop_timeout_env_also_applies_to_implicit_stops(
    task_name, monkeypatch
):
    monkeypatch.setenv("CONTROLLER_STOP_TIMEOUT", "300")

    assert _restart_timeout_used_by(getattr(tasks, task_name)) == [300.0]


def test_terminate_pids_rechecks_before_sigkill():
    """A pid that exits during the last sleep must not be SIGKILLed.

    The in-loop poll can still see the process, and it can exit before
    ``kill -9``. SIGKILL only the pids that are alive at that moment (T8).
    """
    polls_in_loop = int(
        tasks.CONTROLLER_TERMINATE_TIMEOUT_SECONDS
        / tasks.CONTROLLER_TERMINATE_POLL_INTERVAL_SECONDS
    )
    fake_run, calls = _run_recorder()
    # Alive for every poll inside the wait, gone on the re-check.
    pid_alive, polls = _liveness([True] * polls_in_loop + [False])
    monotonic, sleep = _advance_clock()
    with _patched(fake_run, pid_alive, monotonic, sleep):
        tasks._terminate_pids(None, "4242")

    assert not any(cmd.startswith("kill -9") for cmd in calls), calls
    assert len(polls) == polls_in_loop + 1


def test_terminate_pids_multiple_pids_sigkill_only_alive():
    """Only the still-alive pids receive SIGKILL after the grace period."""
    fake_run, calls = _run_recorder()
    alive = {"1": False, "2": True}  # pid 1 exits at once, pid 2 ignores SIGTERM
    monotonic, sleep = _advance_clock()
    with _patched(fake_run, lambda pid: alive[pid], monotonic, sleep):
        tasks._terminate_pids(None, "1 2")

    assert "kill 1 2" in calls
    assert "kill -9 2" in calls
    assert "kill -9 1" not in calls


def test_kill_controller_sends_sigterm_before_sigkill():
    """kill-controller SIGTERMes the Hypercorn pid found via lsof, then SIGKILLs."""
    fake_run, calls = _run_recorder(_LSOF_HYPERCORN)
    pid_alive, _ = _liveness([])  # stays alive -> forces SIGKILL
    monotonic, sleep = _advance_clock()
    with _patched(
        fake_run,
        pid_alive,
        monotonic,
        sleep,
        get_controller_port=mock.Mock(return_value=5000),
    ):
        tasks.kill_controller(_CTX)

    assert "kill 222" in calls
    assert "kill -9 222" in calls
    assert calls.index("kill 222") < calls.index("kill -9 222")


def test_kill_controller_returns_when_controller_exits():
    """kill-controller returns without SIGKILL once the controller exits."""
    fake_run, calls = _run_recorder(_LSOF_HYPERCORN.replace("hypercorn", "python"))
    pid_alive, _ = _liveness([False])  # gone right after SIGTERM
    monotonic, sleep = _advance_clock()
    with _patched(
        fake_run,
        pid_alive,
        monotonic,
        sleep,
        get_controller_port=mock.Mock(return_value=5000),
    ):
        tasks.kill_controller(_CTX)

    assert "kill 222" in calls
    assert not any(cmd.startswith("kill -9") for cmd in calls)


def test_kill_controller_uses_stopping_message():
    """The message no longer says 'Killing previous Hypercorn instances...'."""
    fake_run, _calls = _run_recorder(_LSOF_HYPERCORN)
    pid_alive, _ = _liveness([False])
    monotonic, sleep = _advance_clock()
    messages = []

    def fake_message(msg, level=None):
        messages.append(msg)

    with _patched(
        fake_run,
        pid_alive,
        monotonic,
        sleep,
        get_controller_port=mock.Mock(return_value=5000),
        message=fake_message,
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
    """The 0.5 s poll must not also pay for the invoke spinner's sleep.

    The pid stays alive, so the loop polls until the timeout and then
    SIGKILLs. Only the poll interval may reach sleep().
    """
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
        mock.patch.object(tasks, "_pid_alive", return_value=True),
    ):
        tasks._terminate_pids(SimpleNamespace(run=fake_c_run), "4242")

    interval = tasks.CONTROLLER_TERMINATE_POLL_INTERVAL_SECONDS
    polls = int(tasks.CONTROLLER_TERMINATE_TIMEOUT_SECONDS / interval)
    assert sleeps == [interval] * polls
