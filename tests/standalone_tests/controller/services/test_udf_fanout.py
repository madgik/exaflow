import concurrent.futures
import threading

import grpc
import pytest

from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.controller.services.exareme3.algorithm_flow_engine_interface import (
    Exareme3AlgorithmFlowEngineInterface,
)
from exaflow.controller.services.exareme3.tasks_handler import Exareme3TasksHandler
from exaflow.controller.worker_client.app import WorkerClient
from exaflow.controller.worker_client.app import WorkerClientTimeoutException
from exaflow.protos.worker import worker_pb2
from exaflow.worker_communication import InsufficientDataError


@pytest.fixture(autouse=True)
def _safe_controller_config(monkeypatch):
    from exaflow.controller import config as ctrl_config

    monkeypatch.setattr(ctrl_config, "log_level", "INFO", raising=False)


def _sample_udf():
    return None


class _CallbackCounter:
    """Counts done-callback registrations so tests can synchronize with the fan-out."""

    def __init__(self, target: int):
        self._target = target
        self._count = 0
        self._event = threading.Event()

    def register(self):
        self._count += 1
        if self._count >= self._target:
            self._event.set()

    def wait(self):
        assert self._event.wait(timeout=10), "UDF futures were not fanned out"


class _TrackedFuture(concurrent.futures.Future):
    """A ``concurrent.futures.Future`` that reports when the engine attaches a callback."""

    def __init__(self, counter: _CallbackCounter):
        super().__init__()
        self._counter = counter

    def add_done_callback(self, fn):
        super().add_done_callback(fn)
        self._counter.register()


def _make_handlers(n: int):
    return [
        Exareme3TasksHandler(
            request_id="fanout-req",
            worker_id=f"w{idx}",
            worker_queue_addr=f"127.0.0.1:{6000 + idx}",
            tasks_timeout=30,
        )
        for idx in range(n)
    ]


def _patch_handler_io(handler, future):
    """Point a handler's start/finish at a fake future (no gRPC, no client)."""
    setattr(handler, "start_run_udf", (lambda **_kwargs: future))
    setattr(handler, "finish_run_udf", (lambda future: future.result()))


def _make_engine(handlers):
    return Exareme3AlgorithmFlowEngineInterface(
        request_id="fanout-req",
        context_id="fanout-ctx",
        tasks_handlers=handlers,
        inputdata=Inputdata(
            data_model="data_model", datasets=["dataset"], variables=["x"]
        ),
        metadata={},
    )


def _run_with_driver(engine, futures, drive):
    """Run ``engine.run_udf`` on the main thread while ``drive`` completes futures."""
    counter = _CallbackCounter(len(futures))
    for handler, future in zip(engine._tasks_handlers, futures):
        future._counter = counter
        _patch_handler_io(handler, future)

    holder = {}

    def driver():
        counter.wait()
        drive()

    driver_thread = threading.Thread(target=driver, daemon=True)
    driver_thread.start()
    try:
        holder["result"] = engine.run_udf(_sample_udf, False, False, {})
    except BaseException as exc:  # noqa: BLE001
        holder["exc"] = exc
    driver_thread.join(timeout=10)
    assert not driver_thread.is_alive(), "driver thread leaked"
    return holder


def test_run_udf_returns_results_in_handler_order_when_finishing_out_of_order():
    handlers = _make_handlers(2)
    futures = [_TrackedFuture(_CallbackCounter(2)), _TrackedFuture(_CallbackCounter(2))]

    def drive():
        # Worker w1 finishes before w0.
        futures[1].set_result({"worker": "w1"})
        futures[0].set_result({"worker": "w0"})

    engine = _make_engine(handlers)
    holder = _run_with_driver(engine, futures, drive)

    assert "result" in holder
    assert holder["result"] == [{"worker": "w0"}, {"worker": "w1"}]


def test_run_udf_starts_no_threads():
    handlers = _make_handlers(2)
    futures = [_TrackedFuture(_CallbackCounter(2)), _TrackedFuture(_CallbackCounter(2))]
    futures[0].set_result({"worker": "w0"})
    futures[1].set_result({"worker": "w1"})
    for handler, future in zip(handlers, futures):
        _patch_handler_io(handler, future)

    engine = _make_engine(handlers)
    before = set(threading.enumerate())
    result = engine.run_udf(_sample_udf, False, False, {})
    after = set(threading.enumerate())

    # An unrelated thread may exit meanwhile; only a new one is a failure.
    assert not (after - before)
    assert result == [{"worker": "w0"}, {"worker": "w1"}]


def test_run_udf_skips_worker_reporting_insufficient_data():
    handlers = _make_handlers(2)
    futures = [_TrackedFuture(_CallbackCounter(2)), _TrackedFuture(_CallbackCounter(2))]

    def drive():
        futures[0].set_exception(InsufficientDataError("no rows on w0"))
        futures[1].set_result({"worker": "w1"})

    engine = _make_engine(handlers)
    holder = _run_with_driver(engine, futures, drive)

    assert "result" in holder
    assert holder["result"] == [{"worker": "w1"}]


def test_run_udf_raises_insufficient_data_when_all_report_insufficient():
    handlers = _make_handlers(2)
    futures = [_TrackedFuture(_CallbackCounter(2)), _TrackedFuture(_CallbackCounter(2))]

    def drive():
        futures[0].set_exception(InsufficientDataError("no rows on w0"))
        futures[1].set_exception(InsufficientDataError("no rows on w1"))

    engine = _make_engine(handlers)
    holder = _run_with_driver(engine, futures, drive)

    assert "exc" in holder
    assert isinstance(holder["exc"], InsufficientDataError)
    assert (
        str(holder["exc"])
        == "No workers had sufficient data to run the requested algorithm."
    )


def test_run_udf_cancels_pending_siblings_when_one_worker_fails():
    handlers = _make_handlers(3)
    futures = [
        _TrackedFuture(_CallbackCounter(3)),
        _TrackedFuture(_CallbackCounter(3)),
        _TrackedFuture(_CallbackCounter(3)),
    ]

    def drive():
        # Only w0 fails; w1 and w2 stay pending until the engine cancels them.
        futures[0].set_exception(RuntimeError("boom on w0"))

    engine = _make_engine(handlers)
    holder = _run_with_driver(engine, futures, drive)

    assert "exc" in holder
    assert isinstance(holder["exc"], RuntimeError)
    assert futures[1].cancelled()
    assert futures[2].cancelled()


def test_run_udf_cancels_started_calls_when_starting_a_later_call_fails():
    handlers = _make_handlers(2)
    started_future = _TrackedFuture(_CallbackCounter(1))

    def start_first(**_kwargs):
        return started_future

    def start_second(**_kwargs):
        raise RuntimeError("cannot start w1")

    setattr(handlers[0], "start_run_udf", start_first)
    setattr(handlers[1], "start_run_udf", start_second)
    setattr(handlers[0], "finish_run_udf", (lambda future: future.result()))
    setattr(handlers[1], "finish_run_udf", (lambda future: future.result()))

    engine = _make_engine(handlers)

    with pytest.raises(RuntimeError, match="cannot start w1"):
        engine.run_udf(_sample_udf, False, False, {})

    assert started_future.cancelled()


def test_no_handlers_returns_empty_list():
    engine = _make_engine([])
    assert engine.run_udf(_sample_udf, False, False, {}) == []


class _RpcError(grpc.RpcError):
    def __init__(self, code, details):
        super().__init__(f"{code}: {details}")
        self._code = code
        self._details = details

    def code(self):
        return self._code

    def details(self):
        return self._details


class _FakeFuture:
    def __init__(self, error):
        self._error = error

    def add_done_callback(self, fn):
        # The RPC has already failed, matching grpc.Future for a completed call.
        fn(self)

    def cancel(self):
        return False

    def result(self):
        raise self._error


class _FakeRpcMethod:
    def __init__(self, error):
        self._error = error

    def future(self, request, timeout=None):
        return _FakeFuture(self._error)


class _FakeStub:
    def __init__(self, error):
        self._error = error

    def __getattr__(self, name):
        return _FakeRpcMethod(self._error)


def test_worker_client_call_maps_deadline_exceeded_future_to_timeout_exception():
    client = WorkerClient("127.0.0.1:1")
    try:
        error = _RpcError(grpc.StatusCode.DEADLINE_EXCEEDED, "deadline exceeded")
        setattr(client, "_stub", _FakeStub(error))
        with pytest.raises(WorkerClientTimeoutException):
            client.call("RunUdf", worker_pb2.RunUdfRequest(request_id="req"), timeout=1)
    finally:
        client.close()
