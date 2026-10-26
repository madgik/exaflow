import threading
import time

import pytest

from exaflow.controller.worker_client import app as worker_client_app
from exaflow.controller.worker_client.app import WorkerClientFactory
from exaflow.utils import Singleton


def _drop_worker_client_factory():
    factory = Singleton._instances.pop(WorkerClientFactory, None)
    if factory is None:
        return
    for client in list(factory._clients.values()):
        client.close()


@pytest.fixture(autouse=True)
def reset_worker_client_factory():
    _drop_worker_client_factory()
    yield
    _drop_worker_client_factory()


def test_retain_only_drops_clients_for_departed_workers():
    factory = WorkerClientFactory()
    kept = factory.get_client("10.0.0.1:5670")
    departed = factory.get_client("10.0.0.2:5670")

    factory.retain_only({"10.0.0.1:5670"})

    assert factory.get_client("10.0.0.1:5670") is kept
    assert factory.get_client("10.0.0.2:5670") is not departed
    departed.close()


def test_get_client_builds_one_client_per_address_under_concurrency(monkeypatch):
    factory = WorkerClientFactory()
    constructions = []
    constructing = threading.Event()
    release = threading.Event()

    class BlockingClient:
        def __init__(self, socket_addr):
            constructions.append(socket_addr)
            constructing.set()
            release.wait(5)

        def close(self):
            pass

    monkeypatch.setattr(worker_client_app, "WorkerClient", BlockingClient)

    clients = []
    start = threading.Barrier(2)

    def call_get_client():
        start.wait(5)
        clients.append(factory.get_client("10.0.0.1:5670"))

    threads = [threading.Thread(target=call_get_client) for _ in range(2)]
    for thread in threads:
        thread.start()

    try:
        # Let the second thread reach the cache lookup while the first
        # construction is still in flight. Without the lock both build a client.
        assert constructing.wait(5)
        time.sleep(0.1)
    finally:
        release.set()

    for thread in threads:
        thread.join(5)

    assert constructions == ["10.0.0.1:5670"]
    assert len({id(client) for client in clients}) == 1
