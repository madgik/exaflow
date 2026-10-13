"""Real Quart route, DuckDB loading, protobuf, workers, and aggregation RPCs."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import duckdb
import grpc
import numpy as np
import pytest

import exaflow
from exaflow.aggregation_clients.controller_aggregation_client import (
    ControllerAggregationClient,
)
from exaflow.aggregation_clients.exareme3_udf_aggregation_client import (
    Exareme3UDFAggregationClient,
)
from exaflow.aggregation_serialization import ndarray_to_bytes
from exaflow.aggregation_server.feta import FetaAggregationService
from exaflow.aggregation_server.server import AggregationServer
from exaflow.algorithms.federated.statistics.pearson_correlation import (
    FederatedPearsonCorrelation,
)
from exaflow.controller import config as controller_config
from exaflow.controller import services
from exaflow.feta.clients import FetaAggregationClient
from exaflow.feta.lifecycle import CleanupJob
from exaflow.protos.aggregation_server import aggregation_server_pb2_grpc as legacy_rpc
from exaflow.protos.feta import feta_pb2 as pb
from exaflow.protos.feta import feta_pb2_grpc as rpc
from exaflow.worker import config as worker_config
from exaflow.worker.feta.data_loader import load_algorithm_arrow_table
from exaflow.worker.feta.service import FetaWorkerService
from exaflow.worker_communication import CommonDataElement
from tests.standalone_tests.feta.test_feta_pearson_correlation import (
    IdentityAggregation,
)
from tests.standalone_tests.feta.test_feta_pearson_correlation import data
from tests.standalone_tests.feta.test_feta_ttest_onesample import assert_reference
from tests.standalone_tests.feta.test_feta_ttest_paired import (
    assert_reference as assert_paired_reference,
)


@pytest.fixture
def running_package(tmp_path, monkeypatch):
    frame = data(52)
    frame["group"] = np.resize(["control", "treated", "other"], len(frame))
    frame["dataset"] = ["a"] * 25 + ["b"] * 27
    database = tmp_path / "worker.duckdb"
    connection = duckdb.connect(str(database))
    connection.register("source", frame)
    connection.execute('CREATE TABLE "test_1_0__primary_data" AS SELECT * FROM source')
    connection.close()
    monkeypatch.setitem(worker_config.duckdb, "path", str(database))
    monkeypatch.setitem(controller_config.aggregation_server, "enabled", True)
    monkeypatch.setitem(controller_config, "worker_tasks_timeout", 5)
    monkeypatch.setitem(controller_config.smpc, "enabled", False)
    monkeypatch.setitem(controller_config.smpc, "optional", False)

    servers, jobs, workers, infos = [], [], [], {}

    def start(service, register):
        capacity = (
            service.settings.aggregation_thread_capacity(6)
            if isinstance(service, FetaAggregationService)
            else 6
        )
        pool = ThreadPoolExecutor(max_workers=capacity)
        server = grpc.server(pool)
        register(service, server)
        if isinstance(service, FetaAggregationService):
            legacy_rpc.add_AggregationServerServicer_to_server(
                AggregationServer(), server
            )
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        servers.append((server, pool))
        return port

    aggregation = FetaAggregationService(wait_timeout=2)
    port = start(aggregation, rpc.add_FetaAggregationServicer_to_server)
    aggregation_address = f"127.0.0.1:{port}"
    monkeypatch.setitem(
        controller_config.aggregation_server, "dns", aggregation_address
    )
    for idx, dataset in enumerate(("a", "b")):

        def loader(inputdata, include_dataset=False, dataset=dataset):
            selection = inputdata.model_copy(
                update={"datasets": [dataset] if dataset in inputdata.datasets else []}
            )
            return load_algorithm_arrow_table(
                selection, include_dataset=include_dataset
            )

        service = FetaWorkerService(
            f"worker-{idx}",
            loader=loader,
            aggregation_address=aggregation_address,
            minimum_rows=10,
        )
        port = start(service, rpc.add_FetaWorkerServicer_to_server)
        infos[service.worker_id] = SimpleNamespace(
            id=service.worker_id, ip="127.0.0.1", port=port
        )
        workers.append(service)
        job = CleanupJob(service.store.expire, 0.05)
        job.start()
        jobs.append(job)
    job = CleanupJob(aggregation.expire, 0.05)
    job.start()
    jobs.append(job)

    class Landscape:
        def get_training_and_validation_datasets(self, model):
            assert model == "test:1.0"
            return ["a", "b"], []

        def get_cdes(self, model):
            cdes = {
                column: CommonDataElement(
                    code=column, label=column, sql_type="real", is_categorical=False
                )
                for column in frame
                if column not in ("dataset", "group")
            }
            cdes["group"] = CommonDataElement(
                code="group",
                label="Group",
                sql_type="text",
                is_categorical=True,
                enumerations={
                    label: label for label in ("control", "treated", "other")
                },
            )
            cdes["dataset"] = CommonDataElement(
                code="dataset",
                label="Dataset",
                sql_type="text",
                is_categorical=True,
                enumerations={"a": "a", "b": "b"},
            )
            return cdes

        def get_worker_ids_with_any_of_datasets(self, model, datasets):
            return set(infos)

        def get_worker_info(self, worker_id):
            return infos[worker_id]

    monkeypatch.setattr(services, "_worker_landscape_aggregator", Landscape())
    from exaflow.controller.quart import endpoints
    from exaflow.controller.quart.app import app
    from exaflow.controller.services import (
        algorithm_execution_strategy_factory as factory,
    )
    from exaflow.controller.services.api.algorithm_spec_dtos import (
        _get_algorithm_specifications_dtos,
    )
    from exaflow.controller.services.specifications import Specifications

    specifications = Specifications()
    monkeypatch.setattr(endpoints, "specifications", specifications)
    monkeypatch.setattr(factory, "specifications", specifications)
    monkeypatch.setattr(
        endpoints,
        "algorithm_specifications_dtos",
        _get_algorithm_specifications_dtos(
            list(specifications.enabled_algorithms.values())
        ),
    )
    try:
        yield SimpleNamespace(
            app=app,
            frame=frame,
            workers=workers,
            aggregation=aggregation,
            database=database,
            specifications=specifications,
            aggregation_address=aggregation_address,
        )
    finally:
        for job in jobs:
            job.stop()
        for server, pool in servers:
            server.stop(0).wait()
            pool.shutdown(wait=True)


def request_body(omitted_x=False):
    return {
        "inputdata": {
            "data_model": "test:1.0",
            "datasets": ["a", "b"],
            "variables": ["x0", "x1", "x2", "y0", "y1"],
        },
        "algorithm": {
            "name": "feta_pearson_correlation",
            "x": None if omitted_x else ["x2", "x0", "x1"],
            "y": ["y1", "y0"],
            "parameters": {"alpha": 0.05},
        },
    }


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_one_sample_ttest_through_real_analysis_route(running_package, alternative):
    package = running_package
    parameters = {"mu": 0.25, "alpha": 0.05, "alt_hypothesis": alternative}
    body = {
        "inputdata": {
            "data_model": "test:1.0",
            "datasets": ["a", "b"],
            "variables": ["y0"],
        },
        "algorithm": {
            "name": "feta_ttest_onesample",
            "y": ["y0"],
            "parameters": parameters,
        },
    }

    async def run():
        client = package.app.test_client()
        metadata = await client.get("/specifications/algorithms")
        specifications = json.loads(await metadata.get_data())
        specification = next(
            spec for spec in specifications if spec["name"] == "feta_ttest_onesample"
        )
        assert specification["type"] == "feta"
        assert specification["required_preprocessing"] == []
        response = await client.post("/analysis", json=body)
        payload = await response.get_data()
        assert response.status_code == 200, payload.decode()
        return json.loads(payload)

    result = asyncio.run(run())
    assert_reference(result, package.frame.y0.to_numpy(), parameters)
    assert not package.aggregation.sessions
    assert all(not worker.store.contexts for worker in package.workers)


@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
def test_paired_ttest_through_real_analysis_route(running_package, alternative):
    package = running_package
    parameters = {"alpha": 0.05, "alt_hypothesis": alternative}
    body = {
        "inputdata": {
            "data_model": "test:1.0",
            "datasets": ["a", "b"],
            "variables": ["x0", "y0"],
        },
        "algorithm": {
            "name": "feta_ttest_paired",
            "x": ["x0"],
            "y": ["y0"],
            "parameters": parameters,
        },
    }

    async def run():
        client = package.app.test_client()
        metadata = await client.get("/specifications/algorithms")
        specifications = json.loads(await metadata.get_data())
        specification = next(
            spec for spec in specifications if spec["name"] == "feta_ttest_paired"
        )
        assert specification["type"] == "feta"
        assert specification["required_preprocessing"] == []
        response = await client.post("/analysis", json=body)
        payload = await response.get_data()
        assert response.status_code == 200, payload.decode()
        result = json.loads(payload)
        for group in ("x", "y"):
            invalid = {**body, "algorithm": {**body["algorithm"], group: None}}
            response = await client.post("/analysis", json=invalid)
            assert response.status_code == 460
        return result

    result = asyncio.run(run())
    assert_paired_reference(
        result, package.frame.x0.to_numpy(), package.frame.y0.to_numpy(), parameters
    )
    assert not package.aggregation.sessions
    assert all(not worker.store.contexts for worker in package.workers)


@pytest.mark.parametrize("omitted_x", [False, True])
def test_whole_package_analysis_route_with_real_federation(running_package, omitted_x):
    package = running_package

    async def run():
        client = package.app.test_client()
        metadata = await client.get("/specifications/algorithms")
        specifications = json.loads(await metadata.get_data())
        feta = next(
            spec
            for spec in specifications
            if spec["name"] == "feta_pearson_correlation"
        )
        assert feta["type"] == "feta" and feta["required_preprocessing"] == []
        assert any(spec["name"] == "pearson_correlation" for spec in specifications)
        response = await client.post("/analysis", json=request_body(omitted_x))
        body = await response.get_data()
        assert response.status_code == 200, body.decode()
        return json.loads(body)

    result = asyncio.run(run())
    assert result["n_obs"] == 52
    x_names = ["y1", "y0"] if omitted_x else ["x2", "x0", "x1"]
    assert result["correlations"]["variables"] == x_names
    reference = FederatedPearsonCorrelation(IdentityAggregation()).corrcoef(
        data=package.frame, x_vars=x_names, y_vars=["y1", "y0"], alpha=0.05
    )
    for field in ("correlations", "p_values", "ci_hi", "ci_lo"):
        np.testing.assert_allclose(
            [result[field]["y1"], result[field]["y0"]],
            getattr(reference, field),
            atol=1e-12,
            rtol=1e-10,
            equal_nan=True,
        )
    assert not package.aggregation.sessions
    assert all(not worker.store.contexts for worker in package.workers)
    assert exaflow.get_feta_algorithm_classes() == exaflow.feta_algorithm_classes


def test_real_duckdb_filtering_and_privacy_skip(running_package):
    package = running_package
    body = request_body()
    body["inputdata"]["filters"] = {
        "id": "x0",
        "type": "double",
        "operator": "greater",
        "value": 0,
    }

    async def run():
        response = await package.app.test_client().post("/analysis", json=body)
        payload = await response.get_data()
        assert response.status_code == 200, payload.decode()
        return json.loads(payload)

    result = asyncio.run(run())
    selected = package.frame[package.frame.x0 > 0]
    expected_rows = sum(
        len(part) for _, part in selected.groupby("dataset") if len(part) >= 10
    )
    assert result["n_obs"] == expected_rows
    assert all(not worker.store.contexts for worker in package.workers)


def test_api_rejects_preprocessing_and_missing_values(running_package):
    package = running_package

    async def post(body):
        return await package.app.test_client().post("/analysis", json=body)

    body = request_body()
    body["preprocessing"] = [{"name": "missing_values_handler", "parameters": {}}]
    response = asyncio.run(post(body))
    assert response.status_code == 460
    assert all(not worker.store.contexts for worker in package.workers)
    connection = duckdb.connect(str(package.database))
    connection.execute(
        "UPDATE \"test_1_0__primary_data\" SET x0 = NULL WHERE dataset = 'a'"
    )
    connection.close()
    response = asyncio.run(post(request_body()))
    assert response.status_code == 460
    assert not package.aggregation.sessions
    assert all(not worker.store.contexts for worker in package.workers)


def test_existing_aggregation_protocol_coexists_on_same_server(running_package):
    address = running_package.aggregation_address
    controller = ControllerAggregationClient("legacy-session", aggregator_dns=address)
    clients = [
        Exareme3UDFAggregationClient("legacy-session", aggregator_dns=address)
        for _ in range(2)
    ]
    try:
        controller.configure(num_workers=2)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(client.sum, np.array([idx + 1.0, 3.0]))
                for idx, client in enumerate(clients)
            ]
            for future in futures:
                np.testing.assert_array_equal(future.result(timeout=3), [3.0, 6.0])
        controller.cleanup()
    finally:
        controller.close()
        for client in clients:
            client.close()


def test_cancelled_controller_execution_cleans_up_initialization_in_flight(
    running_package,
):
    from exaflow.controller.services.api.analysis_request_dtos import AnalysisRequestDTO
    from exaflow.controller.services.feta.strategy import FetaStrategy

    package = running_package
    loading, release = Event(), Event()
    original = package.workers[0].loader

    def slow_loader(*args, **kwargs):
        loading.set()
        assert release.wait(3)
        return original(*args, **kwargs)

    package.workers[0].loader = slow_loader
    strategy = FetaStrategy(AnalysisRequestDTO.model_validate(request_body()))

    async def run():
        task = asyncio.create_task(strategy.execute())
        assert await asyncio.to_thread(loading.wait, 2)
        task.cancel()
        try:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=2)
        finally:
            release.set()
        assert strategy.runtime.closed

    asyncio.run(run())
    assert not package.aggregation.sessions
    assert all(not worker.store.contexts for worker in package.workers)


def test_blocking_fetches_leave_submission_threads_available(
    running_package, monkeypatch
):
    service = running_package.aggregation
    started = {f"concurrent-{idx}": Event() for idx in range(8)}
    original = service.fetch

    def fetch(request, active):
        started[request.context_id].set()
        return original(request, active)

    monkeypatch.setattr(service, "fetch", fetch)
    client = FetaAggregationClient(running_package.aggregation_address, timeout=3)
    try:
        for context_id in started:
            client.prepare(
                pb.PrepareRequest(
                    context_id=context_id,
                    step=1,
                    worker_ids=["a"],
                    shape=[1],
                    dtype="float64",
                )
            )
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {key: pool.submit(client.fetch, key, 1) for key in started}
            assert all(event.wait(1) for event in started.values())
            for idx, context_id in enumerate(started):
                client.submit(
                    pb.SubmitRequest(
                        context_id=context_id,
                        step=1,
                        worker_id="a",
                        value=pb.TensorValue(
                            arrow=ndarray_to_bytes(np.array([float(idx)]))
                        ),
                    )
                )
            for idx, context_id in enumerate(started):
                np.testing.assert_array_equal(
                    futures[context_id].result(timeout=1), [float(idx)]
                )
    finally:
        for context_id in started:
            client.cleanup(context_id)
        client.close()


def test_package_skips_filtered_worker_before_aggregation(running_package, monkeypatch):
    package = running_package
    body = request_body()
    body["inputdata"]["filters"] = {
        "id": "dataset",
        "type": "string",
        "operator": "equal",
        "value": "b",
    }

    def unexpected_operation(request, active=lambda: True):
        raise AssertionError("The skipped worker must not execute tensor operations.")

    monkeypatch.setattr(package.workers[0], "execute", unexpected_operation)
    prepared_workers = []
    original = package.aggregation.prepare

    def prepare(request):
        prepared_workers.append(list(request.worker_ids))
        return original(request)

    monkeypatch.setattr(package.aggregation, "prepare", prepare)

    async def run():
        response = await package.app.test_client().post("/analysis", json=body)
        payload = await response.get_data()
        assert response.status_code == 200, payload.decode()
        return json.loads(payload)

    result = asyncio.run(run())
    assert result["n_obs"] == 27
    assert prepared_workers == [["worker-1"]] * 6
    assert all(not worker.store.contexts for worker in package.workers)
    assert not package.aggregation.sessions


@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
def test_independent_ttest_through_real_analysis_route(running_package, alternative):
    from tests.standalone_tests.feta.test_feta_ttest_independent import assert_reference

    package = running_package
    parameters = dict(
        groupA="control", groupB="treated", alpha=0.05, alt_hypothesis=alternative
    )
    body = {
        "inputdata": {
            "data_model": "test:1.0",
            "datasets": ["a", "b"],
            "variables": ["group", "y0"],
        },
        "algorithm": {
            "name": "feta_ttest_independent",
            "x": ["group"],
            "y": ["y0"],
            "parameters": parameters,
        },
    }

    async def run():
        response = await package.app.test_client().post("/analysis", json=body)
        payload = await response.get_data()
        assert response.status_code == 200, payload.decode()
        return json.loads(payload)

    from exaflow.algorithms.exareme3.statistics.ttest_independent import (
        TTestIndependentResult,
    )

    result = TTestIndependentResult(**asyncio.run(run()))
    assert_reference(result, package.frame.rename(columns={"y0": "value"}), parameters)
    assert not package.aggregation.sessions
    assert all(not worker.store.contexts for worker in package.workers)


def test_anova_through_real_analysis_route(running_package):
    from tests.standalone_tests.feta.test_feta_anova_oneway import assert_nested
    from tests.standalone_tests.feta.test_feta_anova_oneway import reference

    package = running_package
    body = {
        "inputdata": {
            "data_model": "test:1.0",
            "datasets": ["a", "b"],
            "variables": ["group", "y0"],
        },
        "algorithm": {
            "name": "feta_anova_oneway",
            "x": ["group"],
            "y": ["y0"],
            "parameters": {},
        },
    }

    async def run():
        client = package.app.test_client()
        response = await client.post("/analysis", json=body)
        payload = await response.get_data()
        assert response.status_code == 200, payload.decode()
        specification = package.specifications.enabled_algorithms["feta_anova_oneway"]
        assert specification.required_preprocessing == []
        assert specification.type.value == "feta"
        return json.loads(payload)

    expected = reference(package.frame.rename(columns={"y0": "value"}))
    expected["anova_table"]["y_label"] = "y0"
    assert_nested(asyncio.run(run()), expected)
    assert not package.aggregation.sessions
    assert all(not worker.store.contexts for worker in package.workers)


@pytest.mark.parametrize("sstype", [1, 2])
def test_twoway_anova_through_real_analysis_route(running_package, sstype):
    from tests.standalone_tests.feta.test_feta_anova_oneway import assert_nested
    from tests.standalone_tests.feta.test_feta_anova_twoway import reference

    package = running_package
    body = {
        "inputdata": {
            "data_model": "test:1.0",
            "datasets": ["a", "b"],
            "variables": ["group", "dataset", "y0"],
        },
        "algorithm": {
            "name": "feta_anova_twoway",
            "x": ["group", "dataset"],
            "y": ["y0"],
            "parameters": {"sstype": sstype},
        },
    }

    async def run():
        client = package.app.test_client()
        response = await client.post("/analysis", json=body)
        payload = await response.get_data()
        assert response.status_code == 200, payload.decode()
        return json.loads(payload)

    expected = reference(
        package.frame.rename(columns={"group": "a", "dataset": "b", "y0": "value"}),
        sstype,
    )
    expected["terms"] = ["group", "dataset", "group:dataset", "Residuals"]
    assert_nested(asyncio.run(run()), expected)
    assert not package.aggregation.sessions
    assert all(not worker.store.contexts for worker in package.workers)
