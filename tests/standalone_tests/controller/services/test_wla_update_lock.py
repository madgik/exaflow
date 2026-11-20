import asyncio
import logging
import threading

import pytest

from exaflow.controller import DeploymentType
from exaflow.controller.http import endpoints
from exaflow.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    DataModelsMetadataPerWorker,
)
from exaflow.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    WorkerLandscapeAggregator,
)
from exaflow.utils import AttrDict


@pytest.fixture
def aggregator():
    return WorkerLandscapeAggregator(
        logger=logging.getLogger("test"),
        update_interval=1000,
        tasks_timeout=1,
        deployment_type=DeploymentType.LOCAL,
        workers=AttrDict({}),
    )


def test_refresh_callback_runs_outside_lock(aggregator):
    refresh, _ = aggregator._claim_refresh()
    callback_acquired_lock = []

    def check_lock(_):
        acquired = aggregator._refresh_lock.acquire(blocking=False)
        callback_acquired_lock.append(acquired)
        if acquired:
            aggregator._refresh_lock.release()

    refresh.add_done_callback(check_lock)
    aggregator._complete_refresh(refresh)

    assert callback_acquired_lock == [True]


def test_http_and_periodic_callers_share_one_refresh_and_can_retry(
    monkeypatch, aggregator
):
    started = threading.Event()
    release = threading.Event()
    attempts = []

    def fetch():
        attempts.append(None)
        started.set()
        if not release.wait(timeout=5):
            raise AssertionError("Test did not release refresh")
        raise RuntimeError("metadata unavailable")

    monkeypatch.setattr(aggregator, "_fetch_workers_metadata", fetch)
    monkeypatch.setattr(
        endpoints, "get_worker_landscape_aggregator", lambda: aggregator
    )
    monkeypatch.setattr(aggregator, "healthcheck", lambda: None)
    periodic_errors = []

    def periodic_update():
        try:
            aggregator.update()
        except RuntimeError as error:
            periodic_errors.append(str(error))

    async def scenario():
        periodic = threading.Thread(target=periodic_update)
        periodic.start()
        try:
            assert await asyncio.to_thread(started.wait, 3)
            cancelled = asyncio.create_task(endpoints.update_wla())
            waiting = [asyncio.create_task(endpoints.update_wla()) for _ in range(40)]
            await asyncio.sleep(0)
            cancelled.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled
            response = await asyncio.wait_for(endpoints.healthcheck(), 2)
            assert response.status_code == 200
            assert len(attempts) == 1
        finally:
            release.set()
            periodic.join(timeout=5)
        results = await asyncio.gather(*waiting, return_exceptions=True)
        assert all(str(result) == "metadata unavailable" for result in results)
        assert periodic_errors == ["metadata unavailable"]
        assert len(attempts) == 1

        with pytest.raises(RuntimeError, match="metadata unavailable"):
            await aggregator.update_async()
        assert len(attempts) == 2

    asyncio.run(scenario())


def test_http_refresh_can_retry_after_executor_submission_fails(
    monkeypatch, aggregator
):
    monkeypatch.setattr(
        endpoints, "get_worker_landscape_aggregator", lambda: aggregator
    )

    def reject_submission(*args):
        raise RuntimeError("executor unavailable")

    async def scenario():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = threading.Event()

        def fetch():
            loop.call_soon_threadsafe(started.set)
            if not release.wait(timeout=5):
                raise AssertionError("Test did not release refresh")
            return [], DataModelsMetadataPerWorker(data_models_metadata_per_worker={})

        monkeypatch.setattr(aggregator, "_fetch_workers_metadata", fetch)
        with monkeypatch.context() as unavailable:
            unavailable.setattr(loop, "run_in_executor", reject_submission)
            with pytest.raises(RuntimeError, match="executor unavailable"):
                await asyncio.wait_for(endpoints.update_wla(), timeout=5)
        assert not aggregator.initial_refresh_complete.is_set()

        retry = asyncio.create_task(endpoints.update_wla())
        try:
            await asyncio.wait_for(started.wait(), timeout=5)
            assert not retry.done()
        finally:
            release.set()
        response = await asyncio.wait_for(retry, timeout=5)
        assert response.status_code == 200
        assert aggregator.initial_refresh_complete.is_set()

    asyncio.run(scenario())
