"""Pin the HTTP contract of every controller route.

The expectations below (status codes, ``Content-Type`` headers and bodies) were
captured from a live run of ``exaflow.controller.quart.app.app`` through
``app.test_client()`` and are asserted here so that any drift in the served
response is caught. No source under ``exaflow/`` is exercised beyond driving the
real routes; the service layer is stubbed via ``monkeypatch`` exactly as in
``test_threaded_endpoint_errors.py``.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest

from exaflow.controller.quart import endpoints
from exaflow.controller.quart.app import app
from exaflow.controller.services.analysis_execution import AnalysisCapacityExceeded
from exaflow.controller.services.api.algorithm_spec_dtos import (
    algorithm_specifications_dtos,
)
from exaflow.controller.services.api.algorithm_spec_dtos import (
    inputdata_specification_dto,
)
from exaflow.controller.services.api.algorithm_spec_dtos import (
    preprocessing_step_specifications_dtos,
)
from exaflow.controller.services.api.analysis_request_validator import BadRequest
from exaflow.controller.services.errors import WorkerTaskTimeoutError
from exaflow.controller.services.errors import WorkerUnresponsiveError
from exaflow.controller.worker_client.app import WorkerMessageTooLargeError
from exaflow.data_filters import FilterError
from exaflow.smpc_cluster_communication import SMPCUsageError
from exaflow.worker_communication import BadUserInput
from exaflow.worker_communication import CommonDataElement
from exaflow.worker_communication import DataModelAttributes
from exaflow.worker_communication import DataModelUnavailable
from exaflow.worker_communication import DatasetUnavailable
from exaflow.worker_communication import InsufficientDataError

HTML_CONTENT_TYPE = "text/html; charset=utf-8"
JSON_CONTENT_TYPE = "application/json"

INSUFFICIENT_DATA_MESSAGE = (
    "The algorithm could not run with the input "
    "provided because there are insufficient data."
)

ANALYSIS_SUCCESS_BODY = '{"result": 1}'

ANALYSIS_REQUEST = {
    "inputdata": {
        "data_model": "data_model",
        "datasets": ["dataset"],
        "variables": ["variable"],
    },
    "algorithm": {"name": "algorithm", "y": ["variable"]},
}


class FakeWorkerLandscapeAggregator:
    """A configurable stand-in for ``WorkerLandscapeAggregator``."""

    def __init__(
        self,
        *,
        datasets=None,
        locations=None,
        variables=None,
        cdes=None,
        attributes=None,
    ):
        self._datasets = datasets or {}
        self._locations = locations or {}
        self._variables = variables or {}
        self._cdes = cdes or {}
        self._attributes = attributes or {}

    def get_all_available_datasets_per_data_model(self):
        return self._datasets

    def get_datasets_locations(self):
        return SimpleNamespace(datasets_locations=self._locations)

    def get_datasets_variables(self):
        return SimpleNamespace(datasets_variables=self._variables)

    def get_cdes_per_data_model(self):
        return SimpleNamespace(data_models_cdes=self._cdes)

    def get_data_models_attributes(self):
        return self._attributes

    def update(self):
        return None

    def healthcheck(self):
        return None


def _raising_worker_landscape_aggregator(error):
    class _Aggregator:
        def get_all_available_datasets_per_data_model(self):
            raise error

    return _Aggregator()


def _flower_controller(inputdata, request_id="req"):
    class _FlowerIORegistry:
        def __init__(self):
            self.result = None
            self.input_reads = 0
            self._request_id = request_id

        def is_current(self, callback_id):
            return self._request_id is not None and callback_id == self._request_id

        def get_inputdata(self):
            self.input_reads += 1
            return inputdata

        async def set_result(self, result):
            self.result = result

    registry = _FlowerIORegistry()

    class _FlowerController:
        flower_execution_info = registry

    controller = _FlowerController()
    controller.registry = registry
    return controller


class _FakeStrategy:
    async def execute(self):
        return ANALYSIS_SUCCESS_BODY


async def _get(path, headers=None):
    async with app.test_client() as client:
        response = await client.get(path, headers=headers)
        return (
            response.status_code,
            response.content_type,
            await response.get_data(as_text=True),
        )


async def _post(path, json_body=None, headers=None):
    async with app.test_client() as client:
        response = await client.post(path, json=json_body, headers=headers)
        return (
            response.status_code,
            response.content_type,
            await response.get_data(as_text=True),
        )


@pytest.fixture
def aggregator(monkeypatch):
    cde = CommonDataElement(
        code="variable",
        label="Variable",
        sql_type="int",
        is_categorical=False,
    )
    fake = FakeWorkerLandscapeAggregator(
        datasets={"data_model": ["dataset"]},
        locations={"data_model": {"dataset": "worker"}},
        variables={"data_model": {"dataset": ["variable"]}},
        cdes={"data_model": SimpleNamespace(values={"variable": cde})},
        attributes={
            "data_model": DataModelAttributes(tags=[], properties={"property": 1})
        },
    )
    monkeypatch.setattr(endpoints, "get_worker_landscape_aggregator", lambda: fake)
    return fake


def test_datasets_route_contract(aggregator):
    status, content_type, body = asyncio.run(_get("/datasets"))

    assert status == 200
    assert content_type == JSON_CONTENT_TYPE
    assert json.loads(body) == {"data_model": ["dataset"]}


def test_datasets_locations_route_contract(aggregator):
    status, content_type, body = asyncio.run(_get("/datasets_locations"))

    assert status == 200
    assert content_type == JSON_CONTENT_TYPE
    assert json.loads(body) == {"data_model": {"dataset": "worker"}}


def test_datasets_variables_route_contract(aggregator):
    status, content_type, body = asyncio.run(_get("/datasets_variables"))

    assert status == 200
    assert content_type == JSON_CONTENT_TYPE
    assert json.loads(body) == {"data_model": {"dataset": ["variable"]}}


def test_cdes_metadata_route_contract(aggregator):
    status, content_type, body = asyncio.run(_get("/cdes_metadata"))
    expected = {
        "data_model": {
            "variable": aggregator._cdes["data_model"].values["variable"].model_dump()
        }
    }

    assert status == 200
    assert content_type == JSON_CONTENT_TYPE
    assert json.loads(body) == expected


def test_data_models_attributes_route_contract(aggregator):
    status, content_type, body = asyncio.run(_get("/data_models_attributes"))
    expected = {"data_model": aggregator._attributes["data_model"].model_dump()}

    assert status == 200
    assert content_type == JSON_CONTENT_TYPE
    assert json.loads(body) == expected


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (
            "/specifications/inputdata",
            inputdata_specification_dto.model_dump_json(),
        ),
        (
            "/specifications/preprocessing",
            preprocessing_step_specifications_dtos.model_dump_json(),
        ),
        (
            "/specifications/algorithms",
            algorithm_specifications_dtos.model_dump_json(),
        ),
    ],
)
def test_specifications_routes_contract(path, expected):
    status, content_type, body = asyncio.run(_get(path))

    assert status == 200
    assert content_type == HTML_CONTENT_TYPE
    assert body == expected


def test_livez_contract():
    status, content_type, body = asyncio.run(_get("/livez"))

    assert status == 200
    assert content_type == HTML_CONTENT_TYPE
    assert body == ""


def test_healthcheck_contract(monkeypatch):
    monkeypatch.setattr(
        endpoints,
        "get_worker_landscape_aggregator",
        lambda: FakeWorkerLandscapeAggregator(),
    )

    status, content_type, body = asyncio.run(_get("/healthcheck"))

    assert status == 200
    assert content_type == HTML_CONTENT_TYPE
    assert body == ""


def test_wla_route_contract(monkeypatch):
    monkeypatch.setattr(
        endpoints,
        "get_worker_landscape_aggregator",
        lambda: FakeWorkerLandscapeAggregator(),
    )

    status, content_type, body = asyncio.run(_post("/wla"))

    assert status == 200
    assert content_type == HTML_CONTENT_TYPE
    assert body == ""


def test_flower_input_route_contract(monkeypatch):
    monkeypatch.setattr(
        endpoints,
        "get_flower_controller",
        lambda: _flower_controller({"inputdata": "value"}),
    )

    status, content_type, body = asyncio.run(
        _get("/flower/input", headers={"X-Request-Id": "req"})
    )

    assert status == 200
    assert content_type == JSON_CONTENT_TYPE
    assert json.loads(body) == {"inputdata": "value"}


def test_flower_result_route_contract(monkeypatch):
    flower = _flower_controller({})
    monkeypatch.setattr(endpoints, "get_flower_controller", lambda: flower)

    status, content_type, body = asyncio.run(
        _post(
            "/flower/result",
            json_body={"result": 1},
            headers={"X-Request-Id": "req"},
        )
    )

    assert status == 200
    assert content_type == JSON_CONTENT_TYPE
    assert json.loads(body) == {"message": "Result set successfully"}
    assert flower.registry.result == {"result": 1}


FLOWER_CALLBACK_MISMATCH = "Flower callback does not belong to the running analysis."


@pytest.mark.parametrize("headers", [None, {"X-Request-Id": "other"}])
def test_flower_input_rejects_a_callback_for_another_request(monkeypatch, headers):
    flower = _flower_controller({"inputdata": "value"})
    monkeypatch.setattr(endpoints, "get_flower_controller", lambda: flower)

    status, content_type, body = asyncio.run(_get("/flower/input", headers=headers))

    assert status == 409
    assert content_type == HTML_CONTENT_TYPE
    assert body == FLOWER_CALLBACK_MISMATCH
    assert flower.registry.input_reads == 0


@pytest.mark.parametrize("headers", [None, {"X-Request-Id": "other"}])
def test_flower_result_rejects_a_callback_for_another_request(monkeypatch, headers):
    flower = _flower_controller({})
    monkeypatch.setattr(endpoints, "get_flower_controller", lambda: flower)

    status, content_type, body = asyncio.run(
        _post("/flower/result", json_body={"result": 1}, headers=headers)
    )

    assert status == 409
    assert content_type == HTML_CONTENT_TYPE
    assert body == FLOWER_CALLBACK_MISMATCH
    assert flower.registry.result is None


def test_flower_callback_rejected_when_no_execution_is_running(monkeypatch):
    flower = _flower_controller({}, request_id=None)
    monkeypatch.setattr(endpoints, "get_flower_controller", lambda: flower)

    input_status, _, input_body = asyncio.run(
        _get("/flower/input", headers={"X-Request-Id": "req"})
    )
    result_status, _, result_body = asyncio.run(
        _post(
            "/flower/result",
            json_body={"result": 1},
            headers={"X-Request-Id": "req"},
        )
    )

    assert input_status == 409
    assert result_status == 409
    assert input_body == FLOWER_CALLBACK_MISMATCH
    assert result_body == FLOWER_CALLBACK_MISMATCH
    assert flower.registry.input_reads == 0
    assert flower.registry.result is None


def test_analysis_success_route_contract(monkeypatch):
    monkeypatch.setattr(endpoints, "get_worker_landscape_aggregator", lambda: object())
    monkeypatch.setattr(endpoints, "validate_analysis_request", lambda **kwargs: None)
    monkeypatch.setattr(
        endpoints, "get_algorithm_execution_strategy", lambda dto: _FakeStrategy()
    )

    status, content_type, body = asyncio.run(
        _post("/analysis", json_body=ANALYSIS_REQUEST)
    )

    assert status == 200
    assert content_type == HTML_CONTENT_TYPE
    assert body == ANALYSIS_SUCCESS_BODY


def test_analysis_malformed_body_returns_400(monkeypatch):
    monkeypatch.setattr(endpoints, "get_worker_landscape_aggregator", lambda: object())
    monkeypatch.setattr(endpoints, "validate_analysis_request", lambda **kwargs: None)
    monkeypatch.setattr(
        endpoints, "get_algorithm_execution_strategy", lambda dto: _FakeStrategy()
    )

    status, content_type, body = asyncio.run(
        _post("/analysis", json_body={"foo": "bar"})
    )

    assert status == 400
    assert content_type == HTML_CONTENT_TYPE
    assert body.startswith("Analysis execution request malformed:")


@pytest.mark.parametrize(
    ("build_error", "status", "expected_body"),
    [
        (lambda: BadRequest("bad request"), 400, lambda error: error.message),
        (lambda: FilterError("filter error"), 400, lambda error: error.message),
        (lambda: BadUserInput("bad user input"), 460, lambda error: error.message),
        (
            lambda: DataModelUnavailable("worker", "data_model"),
            460,
            lambda error: (
                "Data model 'data_model' is not available in worker: 'worker'."
            ),
        ),
        (
            lambda: DatasetUnavailable("worker", "dataset"),
            460,
            lambda error: "Dataset 'dataset' is not available in worker: 'worker'.",
        ),
        (
            lambda: InsufficientDataError("ignored message"),
            461,
            lambda error: INSUFFICIENT_DATA_MESSAGE,
        ),
        (lambda: SMPCUsageError("smpc error"), 462, lambda error: error.message),
        (
            lambda: WorkerMessageTooLargeError("address", "details", 123456),
            413,
            lambda error: error.message,
        ),
        (
            lambda: AnalysisCapacityExceeded(),
            503,
            lambda error: error.message,
        ),
        (
            lambda: WorkerUnresponsiveError(),
            512,
            lambda error: error.message,
        ),
        (
            lambda: WorkerTaskTimeoutError(),
            513,
            lambda error: error.message,
        ),
    ],
)
def test_mapped_exceptions_are_served_with_their_status_and_message(
    monkeypatch, build_error, status, expected_body
):
    error = build_error()
    monkeypatch.setattr(
        endpoints,
        "get_worker_landscape_aggregator",
        lambda: _raising_worker_landscape_aggregator(error),
    )

    returned_status, content_type, body = asyncio.run(_get("/datasets"))

    assert returned_status == status
    assert content_type == HTML_CONTENT_TYPE
    assert body == expected_body(error)
