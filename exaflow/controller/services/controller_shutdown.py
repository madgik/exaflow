"""Tell in-flight controller work that the process is shutting down.

Uvicorn closes the listen socket on SIGTERM, waits timeout_graceful_shutdown,
then cancels request tasks. A client disconnect cancels the request task too.
The flag set here is how those two differ: shutdown cancels registered gRPC
futures and waits only ``shutdown_rpc_cancel_timeout`` for the analysis thread.
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

    The serving process has to be PID 1. Uvicorn does that when ``workers``
    is 1; a supervisor (``workers`` > 1) never sees the signal Kubernetes
    sends. Uvicorn registers the handler with ``signal.signal``, not the
    event loop, so both styles are chained.
    """
    global _signals_installed
    if _signals_installed:
        return
    loop = asyncio.get_running_loop()
    installed = False
    for sig in (signal.SIGTERM, signal.SIGINT):
        handle = getattr(loop, "_signal_handlers", {}).get(sig)
        if handle is not None and not getattr(handle, "_cancelled", True):
            previous = (handle._callback, tuple(handle._args))
            loop.remove_signal_handler(sig)

            def _loop_handler(previous=previous) -> None:
                begin_controller_shutdown()
                callback, args = previous
                callback(*args)

            loop.add_signal_handler(sig, _loop_handler)
            installed = True
            continue

        previous_signal = signal.getsignal(sig)
        if not callable(previous_signal):
            continue

        def _signal_handler(signum, frame, previous=previous_signal) -> None:
            begin_controller_shutdown()
            previous(signum, frame)

        signal.signal(sig, _signal_handler)
        installed = True
    _signals_installed = installed
