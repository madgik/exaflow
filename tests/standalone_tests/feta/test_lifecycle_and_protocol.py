import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
import pytest

from exaflow.aggregation_serialization import ndarray_to_bytes
from exaflow.aggregation_server.feta import FetaAggregationService
from exaflow.feta.lifecycle import CleanupJob
from exaflow.feta.lifecycle import ContextClosedError
from exaflow.feta.lifecycle import ContextStore
from exaflow.feta.lifecycle import ResourceLimitError
from exaflow.feta.operations import LocalOperation
from exaflow.feta.operations import LocalTensorRef
from exaflow.feta.operations import OperationType
from exaflow.feta.operations import TensorMetadata
from exaflow.feta.operations import validate_array
from exaflow.feta.settings import FetaSettings
from exaflow.protos.feta import feta_pb2 as pb
from tests.standalone_tests.feta.helpers import federation
from tests.standalone_tests.feta.test_feta_pearson_correlation import data
from tests.standalone_tests.feta.test_feta_pearson_correlation import inputs


def test_context_expiry_and_late_results_do_not_recreate_context():
    now = [0.0]
    store = ContextStore(FetaSettings(context_lifetime_minutes=1), clock=lambda: now[0])
    value = np.ones((10, 2))
    metadata = TensorMetadata(value.shape, "float64", 0, "ctx")
    refs = store.create("ctx", {"x": (value, metadata)})
    ctx, _ = store.acquire("ctx", [refs["x"]])
    now[0] = 59
    store.put("ctx", ctx, value, metadata)
    now[0] = 60
    assert store.expire() == 1
    with pytest.raises(ContextClosedError):
        store.put("ctx", ctx, value, metadata)
    assert not store.contexts
    store.cleanup("ctx")


def test_concurrent_context_creation_cleanup_and_publication():
    store = ContextStore(FetaSettings(max_contexts=100))

    def work(idx):
        value = np.ones((2, 2))
        metadata = TensorMetadata(value.shape, "float64", 0, str(idx))
        refs = store.create(str(idx), {"x": (value, metadata)})
        ctx, values = store.acquire(str(idx), [refs["x"]])
        assert values[0][0].sum() == 4
        store.cleanup(str(idx))
        with pytest.raises(ContextClosedError):
            store.put(str(idx), ctx, value, metadata)

    with ThreadPoolExecutor(max_workers=10) as pool:
        list(pool.map(work, range(100)))
    assert not store.contexts


def test_cleanup_job_starts_stops_and_reports_failure():
    scanned = Event()
    job = CleanupJob(scanned.set, 0.005)
    job.start()
    assert scanned.wait(1)
    job.stop()
    assert not job.thread.is_alive()
    failed = Event()

    def fail():
        raise RuntimeError("sweeper failure")

    job = CleanupJob(fail, 0.005, failed.set)
    job.start()
    assert failed.wait(1)
    job.stop()


@pytest.mark.parametrize(
    "message",
    [
        pb.LocalOperation(type=1000),
        pb.LocalOperation(type=pb.ADD, inputs=[pb.Operand()]),
        pb.LocalOperation(type=pb.ADD, inputs=[pb.Operand(tensor_id="")]),
        pb.LocalOperation(
            type=pb.MULTIPLY,
            inputs=[pb.Operand(array=pb.TensorValue(arrow=b"pickle is not Arrow"))],
        ),
    ],
)
def test_malicious_operation_payloads_are_rejected(message):
    with pytest.raises(ValueError):
        LocalOperation.from_proto(message, 1024)


def test_worker_rejects_cross_context_refs_bad_axes_and_raw_export():
    frame = data(20)
    runtime, workers, _ = federation([frame])
    with runtime:
        x, y = runtime.initialize(inputs(frame), ["x0"], ["y0"])
        service = workers[0].service
        ref = LocalTensorRef(x.descriptors["worker-0"].tensor_id)
        with pytest.raises(ValueError, match="Unknown tensor"):
            service.execute(
                pb.ExecuteRequest(
                    context_id=runtime.context_id,
                    operation=LocalOperation(
                        OperationType.MULTIPLY, (LocalTensorRef("other-context-id"), 2)
                    ).to_proto(),
                )
            )
        with pytest.raises(ValueError, match="axis"):
            service.execute(
                pb.ExecuteRequest(
                    context_id=runtime.context_id,
                    operation=LocalOperation(
                        OperationType.SUM, (ref,), axis=10
                    ).to_proto(),
                )
            )
        with pytest.raises(ValueError, match="placement"):
            service.execute(
                pb.ExecuteRequest(
                    context_id=runtime.context_id,
                    operation=LocalOperation(
                        OperationType.TRANSPOSE, (ref,)
                    ).to_proto(),
                    aggregation_step=1,
                )
            )
        with pytest.raises(ValueError, match="Literal operands"):
            x + np.ones((20, 1))
        with pytest.raises(ValueError, match="number"):
            service.execute(
                pb.ExecuteRequest(
                    context_id=runtime.context_id,
                    operation=LocalOperation(OperationType.MULTIPLY, (ref,)).to_proto(),
                )
            )


def prepare(service, context_id="ctx", step=1, shape=(2,), workers=("a", "b")):
    service.prepare(
        pb.PrepareRequest(
            context_id=context_id,
            step=step,
            worker_ids=workers,
            shape=shape,
            dtype="float64",
        )
    )


def submit(service, worker_id, value, context_id="ctx", step=1):
    service.submit(
        pb.SubmitRequest(
            context_id=context_id,
            step=step,
            worker_id=worker_id,
            value=pb.TensorValue(
                arrow=ndarray_to_bytes(np.asarray(value, dtype=float))
            ),
        )
    )


def test_aggregation_submit_ack_and_result_retention_sequence():
    service = FetaAggregationService(wait_timeout=1)
    prepare(service)
    submit(service, "b", [3, 4])  # Returns immediately, before the other worker.
    submit(service, "a", [1, 2])
    np.testing.assert_array_equal(
        service.fetch(pb.StepRequest(context_id="ctx", step=1)), [4, 6]
    )
    with pytest.raises(ValueError, match="sequential"):
        prepare(service, step=3)
    prepare(service, step=2)
    submit(service, "a", [2, 3], step=2)
    submit(service, "b", [4, 5], step=2)
    np.testing.assert_array_equal(
        service.fetch(pb.StepRequest(context_id="ctx", step=2)), [6, 8]
    )
    service.cleanup("ctx")
    service.cleanup("ctx")
    assert not service.sessions


def test_cleanup_and_expiry_wake_blocked_fetch():
    for expire in (False, True):
        now = [0.0]
        service = FetaAggregationService(wait_timeout=60, clock=lambda: now[0])
        prepare(service)
        started = Event()

        def fetch():
            started.set()
            return service.fetch(pb.StepRequest(context_id="ctx", step=1))

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(fetch)
            assert started.wait(1)
            if expire:
                now[0] = 3600
                service.expire()
            else:
                service.cleanup("ctx")
            with pytest.raises(ContextClosedError):
                future.result(timeout=1)
        assert not service.sessions


def test_timeout_and_duplicate_contribution_fail_pending_fetch():
    service = FetaAggregationService(wait_timeout=0.01)
    prepare(service)
    with pytest.raises(TimeoutError):
        service.fetch(pb.StepRequest(context_id="ctx", step=1))
    service.cleanup("ctx")
    prepare(service)
    submit(service, "a", [1, 2])
    with pytest.raises(ValueError, match="Duplicate"):
        submit(service, "a", [3, 4])
    with pytest.raises(ValueError, match="Duplicate"):
        service.fetch(pb.StepRequest(context_id="ctx", step=1))


def test_resource_limits_before_allocation_and_context_creation():
    settings = FetaSettings(max_tensor_bytes=32, max_context_bytes=64)
    service = FetaAggregationService(settings)
    with pytest.raises(ResourceLimitError):
        prepare(service, shape=(1000000,))
    assert not service.sessions
    frame = data(20)
    runtime, workers, _ = federation([frame], settings=settings)
    with pytest.raises(ResourceLimitError):
        with runtime:
            runtime.initialize(inputs(frame), ["x0"], ["y0"])
    assert not workers[0].service.store.contexts


def test_cleanup_during_initial_load_prevents_late_publication():
    frame = data(20)
    runtime, workers, _ = federation([frame])
    service = workers[0].service
    original = service.loader
    loading, finish = Event(), Event()

    def loader(*args, **kwargs):
        loading.set()
        assert finish.wait(1)
        return original(*args, **kwargs)

    service.loader = loader
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(runtime.initialize, inputs(frame), ["x0"], ["y0"])
        assert loading.wait(1)
        service.store.cleanup(runtime.context_id)
        finish.set()
        with pytest.raises(ContextClosedError):
            future.result(timeout=1)
    runtime.close()
    assert not service.store.contexts


def test_failed_operation_is_not_retried_and_contexts_are_cleaned():
    frame = data(40)
    runtime, workers, aggregation = federation([frame.iloc[:20], frame.iloc[20:]])
    count = [0]

    def failure(request):
        count[0] += 1
        raise RuntimeError("worker failed")

    with pytest.raises(RuntimeError, match="worker failed"):
        with runtime:
            x, y = runtime.initialize(inputs(frame), ["x0"], ["y0"])
            workers[1].execute = failure
            x.sum(axis=0)
    assert count[0] == 1
    assert not aggregation.rounds
    assert all(not worker.service.store.contexts for worker in workers)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), 1e308, True])
def test_invalid_lifetime_cannot_disable_expiry(value):
    with pytest.raises(ValueError):
        FetaSettings(context_lifetime_minutes=value)


def test_completed_aggregation_rejects_duplicate_before_fetch():
    service = FetaAggregationService(wait_timeout=1)
    prepare(service)
    submit(service, "a", [1, 2])
    submit(service, "b", [3, 4])
    with pytest.raises(ValueError, match="Duplicate"):
        submit(service, "a", [100, 200])
    with pytest.raises(ValueError, match="Duplicate"):
        service.fetch(pb.StepRequest(context_id="ctx", step=1))


def test_tensor_inputs_are_reused_but_not_returned_as_local_data():
    frame = data(20)
    runtime, workers, _ = federation([frame])
    with runtime:
        x, y = runtime.initialize(inputs(frame), ["x0"], ["x0"])
        assert (
            x.descriptors["worker-0"].tensor_id == y.descriptors["worker-0"].tensor_id
        )
        squared = x * x
        np.testing.assert_allclose(
            squared.sum(axis=0), (frame[["x0"]].to_numpy() ** 2).sum(axis=0)
        )
        np.testing.assert_allclose(
            (squared + squared).sum(axis=0),
            2 * (frame[["x0"]].to_numpy() ** 2).sum(axis=0),
        )
        assert workers[0].operations.count(pb.MULTIPLY) == 1


def test_distinct_contexts_cannot_be_combined():
    frame = data(20)
    first, _, _ = federation([frame])
    second, _, _ = federation([frame])
    with first, second:
        x, y = first.initialize(inputs(frame), ["x0"], ["y0"])
        other_x, _ = second.initialize(inputs(frame), ["x0"], ["y0"])
        with pytest.raises(ValueError, match="different contexts"):
            x + other_x


def test_controller_fetch_begins_while_worker_computation_is_running():
    frame = data(20)
    runtime, workers, aggregation = federation([frame])
    original = workers[0].execute

    def execute(request):
        assert aggregation.fetch_started.wait(1)
        return original(request)

    with runtime:
        x, _ = runtime.initialize(inputs(frame), ["x0"], ["y0"])
        workers[0].execute = execute
        np.testing.assert_allclose(x.sum(axis=0), frame[["x0"]].to_numpy().sum(axis=0))


def test_expired_controller_context_rejects_new_operations():
    frame = data(20)
    runtime, workers, aggregation = federation([frame])
    with runtime:
        x, _ = runtime.initialize(inputs(frame), ["x0"], ["y0"])
        runtime.deadline = 0
        with pytest.raises(ContextClosedError, match="expired"):
            x.sum(axis=0)
        assert not aggregation.events
    assert all(not worker.service.store.contexts for worker in workers)


def test_duplicate_pending_fetch_is_rejected_without_poisoning_first_fetch():
    service = FetaAggregationService(wait_timeout=1)
    prepare(service)
    request = pb.StepRequest(context_id="ctx", step=1)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(service.fetch, request)
        deadline = time.monotonic() + 1
        while not service.sessions["ctx"].fetching:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        with pytest.raises(ValueError, match="Only one"):
            service.fetch(request)
        submit(service, "a", [1, 2])
        submit(service, "b", [3, 4])
        np.testing.assert_array_equal(first.result(timeout=1), [4, 6])


@pytest.mark.parametrize("dtype", ["bool", "object", "complex128", "float16", ">f8"])
def test_unsupported_literal_and_aggregation_dtypes_are_rejected(dtype):
    with pytest.raises(ValueError):
        validate_array(np.zeros(2, dtype=dtype), 1024)
    service = FetaAggregationService()
    with pytest.raises(ValueError):
        service.prepare(
            pb.PrepareRequest(
                context_id="ctx", step=1, worker_ids=["a"], shape=[2], dtype=dtype
            )
        )
    assert not service.sessions


def test_cancelled_initialization_never_loads_or_allocates_data():
    runtime, workers, _ = federation([data(20)])
    service = workers[0].service
    request = pb.InitializeRequest(context_id="cancelled")
    with pytest.raises(ContextClosedError, match="cancelled"):
        service.initialize(request, active=lambda: False)
    assert not service.store.contexts
    runtime.close()


def test_cancellation_after_computation_prevents_storing_result(monkeypatch):
    from exaflow.worker.feta import service as module

    frame = data(20)
    runtime, workers, _ = federation([frame])
    active = [True]
    original = module.evaluate

    def evaluate(*args):
        value = original(*args)
        active[0] = False
        return value

    monkeypatch.setattr(module, "evaluate", evaluate)
    with runtime:
        x, _ = runtime.initialize(inputs(frame), ["x0"], ["y0"])
        service = workers[0].service
        ctx = service.store.contexts[runtime.context_id]
        count = len(ctx.values)
        request = pb.ExecuteRequest(
            context_id=runtime.context_id,
            operation=LocalOperation(
                OperationType.MULTIPLY,
                (LocalTensorRef(x.descriptors["worker-0"].tensor_id), 2),
            ).to_proto(),
        )
        with pytest.raises(ContextClosedError, match="cancelled"):
            service.execute(request, active=lambda: active[0])
        assert len(ctx.values) == count
