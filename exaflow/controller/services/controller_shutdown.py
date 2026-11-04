"""Shutdown budget and the shutdown flag for in-flight controller work.

Hypercorn closes the listen socket on SIGTERM, drains requests for
``GRACEFUL_TIMEOUT`` and then cancels the request tasks. Quart does not pass
that cancellation on to the task that runs the request, so the analysis threads
are still running when the lifespan shutdown hook starts. That hook calls
``begin_controller_shutdown``: it cancels the registered worker and aggregation
RPCs, and every RPC registered afterwards, so a thread that starts its next
batch stops too.

Only the RPCs that run an analysis are registered (UDF calls and aggregation
Configure). Cleanup and metadata RPCs are not, so shutdown cannot abort the
calls that release resources.
"""

import threading
from typing import Any

# Seconds. The pod grace period is the sum of these, the preStop delay in the
# chart, and the margin. It does not include worker_tasks_timeout, because
# shutdown cancels the calls that would wait for it.
GRACEFUL_TIMEOUT = 120  # Hypercorn drains in-flight requests this long
SHUTDOWN_TIMEOUT = 60  # Hypercorn waits this long for the shutdown hooks
SHUTDOWN_RPC_CANCEL_TIMEOUT = 5  # wait for analysis threads after cancelling RPCs
SHUTDOWN_CLEANUP_TIMEOUT = 10  # aggregation cleanup and worker-landscape join
SHUTDOWN_GRACE_MARGIN = 20

_shutting_down = threading.Event()
_inflight_guard = threading.Lock()
_inflight: set[Any] = set()


def begin_controller_shutdown() -> None:
    """Mark shutdown and cancel every registered RPC."""
    with _inflight_guard:
        _shutting_down.set()
        futures = list(_inflight)
    for future in futures:
        future.cancel()


def controller_is_shutting_down() -> bool:
    return _shutting_down.is_set()


def reset_controller_shutdown() -> None:
    """Clear the flag and the in-flight set. Tests use this."""
    _shutting_down.clear()
    with _inflight_guard:
        _inflight.clear()


def _discard(future: Any) -> None:
    with _inflight_guard:
        _inflight.discard(future)


def register_inflight_rpc(future: Any) -> None:
    """Let shutdown cancel future. Cancel it now if shutdown already began."""
    with _inflight_guard:
        shutting_down = _shutting_down.is_set()
        if not shutting_down:
            _inflight.add(future)
    if shutting_down:
        future.cancel()
        return
    future.add_done_callback(_discard)
