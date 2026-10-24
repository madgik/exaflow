"""Bounded off-loop execution for analyses.

Analyses share one thread pool sized to ``max_concurrent_experiments``.
A semaphore of the same size rejects extra requests instead of queueing
them on that pool. Health checks and worker-landscape updates stay on
the default executor.
"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from exaflow.controller import config as ctrl_config

_DEFAULT_MAX_CONCURRENT_EXPERIMENTS = 32
_THREAD_NAME_PREFIX = "exaflow-analysis"

_executor: ThreadPoolExecutor | None = None
_semaphore: asyncio.Semaphore | None = None
_semaphore_loop: asyncio.AbstractEventLoop | None = None
_state_guard = threading.Lock()


class AnalysisCapacityExceeded(Exception):
    def __init__(self) -> None:
        self.message = (
            "The controller is already running the maximum number of "
            "concurrent analyses."
        )
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


async def run_blocking(func, *args):
    """Run func on the analysis pool and wait for it, even if cancelled.

    Cancellation (client disconnect) must not abandon the thread. The caller
    waits until func returns before running cleanup that the thread still needs.
    """
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(_ensure_executor(), func, *args)
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        await future
        raise
