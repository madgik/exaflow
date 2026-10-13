from exaflow.controller.worker_client.app import WorkerClientFactory


def test_retain_only_drops_clients_for_departed_workers():
    factory = WorkerClientFactory()
    kept = factory.get_client("10.0.0.1:5670")
    factory.get_client("10.0.0.2:5670")

    factory.retain_only({"10.0.0.1:5670"})

    assert factory.get_client("10.0.0.1:5670") is kept
    assert set(factory._clients) == {"10.0.0.1:5670"}
