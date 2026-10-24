import logging
import threading
import time

from exaflow.controller import DeploymentType
from exaflow.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    WorkerLandscapeAggregator,
)
from exaflow.utils import AttrDict


def test_update_does_not_overlap_itself(monkeypatch):
    aggregator = WorkerLandscapeAggregator(
        logger=logging.getLogger("test"),
        update_interval=1000,
        tasks_timeout=1,
        deployment_type=DeploymentType.LOCAL,
        localworkers=AttrDict({}),
    )
    state = {"current": 0, "max": 0}
    guard = threading.Lock()

    def fetch(self):
        with guard:
            state["current"] += 1
            state["max"] = max(state["max"], state["current"])
        time.sleep(0.15)
        with guard:
            state["current"] -= 1
        raise RuntimeError("stop after the critical section")

    monkeypatch.setattr(WorkerLandscapeAggregator, "_fetch_workers_metadata", fetch)

    errors = []

    def run():
        try:
            aggregator.update()
        except RuntimeError as exc:
            errors.append(str(exc))

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert state["max"] == 1, state
    assert len(errors) == 2
