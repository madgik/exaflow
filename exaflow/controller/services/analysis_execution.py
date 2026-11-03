"""Bounded off-loop execution for analyses.

Analyses share one thread pool sized to ``max_concurrent_experiments``.
A semaphore of the same size rejects extra requests instead of queueing
them on that pool. Health checks and worker-landscape updates stay on
the default executor.
"""

import asyncio
import contextlib
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from exaflow.controller import config as ctrl_config
from exaflow.controller.http.server_config import shutdown_rpc_cancel_timeout
from exaflow.controller.services.controller_shutdown import cancel_inflight_rpcs
from exaflow.controller.services.controller_shutdown import controller_is_shutting_down

_LOGGER = logging.getLogger(__name__)

_DEFAULT_MAX_CONCURRENT_EXPERIMENTS = 32
_THREAD_NAME_PREFIX = "exaflow-analysis"
DEFAULT_CAPACITY_MESSAGE = (
    "The controller is already running the maximum number of concurrent analyses."
)

_executor: ThreadPoolExecutor | None = None
_semaphore: asyncio.Semaphore | None = None
_semaphore_loop: asyncio.AbstractEventLoop | None = None
_state_guard = threading.Lock()


class AnalysisCapacityExceeded(Exception):
    def __init__(self, message: str = DEFAULT_CAPACITY_MESSAGE) -> None:
        self.message = message
        super().__init__(self.message)


def max_concurrent_experiments() -> int:
    raw = getattr(ctrl_config, "max_concurrent_experiments", None)
    if raw is None or raw == "":
        return _DEFAULT_MAX_CONCURRENT_EXPERIMENTS
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "max_concurrent_experiments must be a positive integer"
        ) from exc
    if value < 1:
        raise RuntimeError("max_concurrent_experiments must be a positive integer")
    return value


def start_analysis_execution() -> None:
    _ensure_executor()
    _ensure_semaphore()


def shutdown_analysis_execution(wait: bool = False) -> None:
    global _executor, _semaphore, _semaphore_loop
    with _state_guard:
        executor = _executor
        _executor = None
        _semaphore = None
        _semaphore_loop = None
    if executor is not None:
        executor.shutdown(wait=wait, cancel_futures=False)


def reset_analysis_execution() -> None:
    """Drop the cached pool and semaphore. Tests use this between limits."""
    shutdown_analysis_execution(wait=False)


def _ensure_executor() -> ThreadPoolExecutor:
    global _executor
    with _state_guard:
        if _executor is None:
            _executor = ThreadPoolExecutor(
                max_workers=max_concurrent_experiments(),
                thread_name_prefix=_THREAD_NAME_PREFIX,
            )
        return _executor


def _ensure_semaphore() -> asyncio.Semaphore:
    global _semaphore, _semaphore_loop
    loop = asyncio.get_running_loop()
    if _semaphore is None or _semaphore_loop is not loop:
        _semaphore = asyncio.Semaphore(max_concurrent_experiments())
        _semaphore_loop = loop
    return _semaphore


@asynccontextmanager
async def analysis_slot():
    """Hold one analysis slot, or reject when every slot is taken."""
    semaphore = _ensure_semaphore()
    if semaphore.locked():
        raise AnalysisCapacityExceeded()
    await semaphore.acquire()
    try:
        yield
    finally:
        semaphore.release()


async def _wait_for_executor_future(future, timeout: float) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not future.done():
        remaining = deadline - loop.time()
        if remaining <= 0:
            break
        try:
            await asyncio.wait_for(asyncio.shield(future), remaining)
        except asyncio.CancelledError:
            continue
        except Exception:
            # The thread finished by failing, including a cancelled RPC.
            return
    if future.done():
        # The thread may have failed because its RPC was cancelled. Retrieve
        # that exception so asyncio does not warn about it.
        future.exception()


async def run_blocking(func, *args):
    """Run func on the analysis pool and wait for it, even if cancelled.

    Cancellation (client disconnect) must not abandon the thread. The caller
    waits until func returns before running cleanup that the thread still needs,
    also when it is cancelled again while waiting.

    Process shutdown is different: in-flight RPCs are cancelled and the wait
    is bounded by shutdown_rpc_cancel_timeout, so the pod grace period does
    not have to cover worker_tasks_timeout.
    """
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(_ensure_executor(), func, *args)
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        if controller_is_shutting_down():
            cancel_inflight_rpcs()
            await _wait_for_executor_future(future, shutdown_rpc_cancel_timeout)
            if not future.done():
                _LOGGER.warning(
                    "Analysis thread still running %s seconds after shutdown "
                    "cancelled its RPCs",
                    shutdown_rpc_cancel_timeout,
                )
            raise
        while not future.done():
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait([future])
        raise
