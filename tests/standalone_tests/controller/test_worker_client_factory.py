import pytest

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
