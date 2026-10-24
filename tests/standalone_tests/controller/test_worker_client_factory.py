import threading
import time

from exaflow.controller.worker_client.app import WorkerClientFactory
from exaflow.utils import Singleton


def test_get_client_does_not_create_duplicates_under_threads(monkeypatch):
    created = []
    state = {"current": 0, "max": 0}
    guard = threading.Lock()

    class FakeWorkerClient:
        def __init__(self, socket_addr):
            with guard:
                state["current"] += 1
                state["max"] = max(state["max"], state["current"])
            time.sleep(0.05)
            with guard:
                state["current"] -= 1
                created.append(socket_addr)
            self.socket_addr = socket_addr

    monkeypatch.setattr(
        "exaflow.controller.worker_client.app.WorkerClient", FakeWorkerClient
    )
    Singleton._instances.pop(WorkerClientFactory, None)
    try:
        factory = WorkerClientFactory()
        threads = [
            threading.Thread(target=factory.get_client, args=("127.0.0.1:5672",))
            for _ in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        Singleton._instances.pop(WorkerClientFactory, None)

    assert created == ["127.0.0.1:5672"]
    assert state["max"] == 1
