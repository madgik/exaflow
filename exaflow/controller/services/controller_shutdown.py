"""Tell in-flight controller work that the process is shutting down.

Hypercorn closes the listen socket on SIGTERM, waits graceful_timeout, then
cancels request tasks and still waits for them to finish. A client disconnect
cancels the request task too. The flag set here is how those two differ:
shutdown cancels registered gRPC futures and waits only
``shutdown_rpc_cancel_timeout`` for the analysis thread.
"""

import asyncio
import signal
import threading
from typing import Any

_shutting_down = threading.Event()
_inflight_guard = threading.Lock()
_inflight: set[Any] = set()
_signals_installed = False


def begin_controller_shutdown() -> None:
    _shutting_down.set()


def controller_is_shutting_down() -> bool:
    return _shutting_down.is_set()


def reset_controller_shutdown() -> None:
    """Clear the flag and the in-flight set. Tests use this."""
    global _signals_installed
    _shutting_down.clear()
    with _inflight_guard:
        _inflight.clear()
    _signals_installed = False


def register_inflight_rpc(future: Any) -> None:
    with _inflight_guard:
        _inflight.add(future)

    def _discard(done: Any) -> None:
        with _inflight_guard:
            _inflight.discard(done)

    future.add_done_callback(_discard)


def cancel_inflight_rpcs() -> None:
    with _inflight_guard:
        futures = list(_inflight)
    for future in futures:
        future.cancel()


def install_shutdown_signal_handlers() -> None:
    """Chain the process SIGTERM/SIGINT handlers so shutdown is visible.

    Hypercorn installs those handlers before the app starts, and only one
    handler can be registered per signal. The serving process has to be the
    one that receives SIGTERM (``workers = 0``); a supervisor child never
    sees the signal Kubernetes sends to PID 1.
    """
    global _signals_installed
    if _signals_installed:
        return
    loop = asyncio.get_running_loop()
    installed = False
    for sig in (signal.SIGTERM, signal.SIGINT):
        handle = getattr(loop, "_signal_handlers", {}).get(sig)
        if handle is None or getattr(handle, "_cancelled", True):
            continue
        previous = (handle._callback, tuple(handle._args))
        loop.remove_signal_handler(sig)

        def _handler(previous=previous) -> None:
            begin_controller_shutdown()
            callback, args = previous
            callback(*args)

        loop.add_signal_handler(sig, _handler)
        installed = True
    _signals_installed = installed
