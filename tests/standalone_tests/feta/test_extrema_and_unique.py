import numpy as np
import pandas as pd
import pytest

from exaflow.aggregation_server.feta import FetaAggregationService
from exaflow.algorithms.feta import FedTensor
from exaflow.algorithms.feta import federated_numpy as fnp
from exaflow.feta.inputdata import Inputdata
from exaflow.feta.lifecycle import ResourceLimitError
from exaflow.feta.operations import decode_tensor
from exaflow.feta.operations import encode_tensor
from exaflow.feta.settings import FetaSettings
from exaflow.protos.feta import feta_pb2 as pb
from tests.standalone_tests.feta.helpers import federation


@pytest.mark.parametrize("function", [fnp.min, fnp.max])
@pytest.mark.parametrize("axis", [None, 0, 1])
def test_extrema_follow_numpy_and_partitioning(function, axis):
    frame = pd.DataFrame({"a": [3.0, -7.0, 2.0, 9.0], "b": [1.0, 4.0, -8.0, 2.0]})
    runtime, _, _ = federation([frame.iloc[:0], frame.iloc[:2], frame.iloc[2:]])
    selection = Inputdata(data_model="test:1.0", datasets=["a"], variables=list(frame))
    with runtime:
        y = runtime.initialize(selection, {"y": list(frame)})["y"]
        result = function(y, axis=axis)
        expected = getattr(np, function.__name__)(frame.to_numpy(), axis=axis)
        if axis == 1:
            assert isinstance(result, FedTensor)
            assert result.shape == (4,)
            assert fnp.sum(result) == expected.sum()
            assert runtime.step == 1  # local extrema, then the explicit sum
        else:
            np.testing.assert_array_equal(result, expected)
            assert runtime.step == 1


@pytest.mark.parametrize("function", [fnp.min, fnp.max])
def test_transposed_and_empty_extrema(function):
    frame = pd.DataFrame({"a": [3.0, -7.0, 2.0, 9.0], "b": [1.0, 4.0, -8.0, 2.0]})
    selection = Inputdata(data_model="test:1.0", datasets=["a"], variables=list(frame))
    runtime, _, _ = federation([frame.iloc[:0], frame.iloc[:2], frame.iloc[2:]])
    with runtime:
        y = runtime.initialize(selection, {"y": list(frame)})["y"]
        np.testing.assert_array_equal(
            function(y.T, axis=1), function(frame.to_numpy().T, axis=1)
        )
    runtime, _, _ = federation([frame.iloc[:0]])
    with runtime:
        y = runtime.initialize(selection, {"y": list(frame)})["y"]
        with pytest.raises(ValueError, match="Zero-size"):
            function(y)
        assert runtime.step == 0


def test_unique_sorted_labels_and_bounded_release():
    frame = pd.DataFrame({"group": ["λ", "a", "a", "long label"]})
    runtime, _, _ = federation([frame.iloc[:0], frame.iloc[:2], frame.iloc[2:]])
    selection = Inputdata(data_model="test:1.0", datasets=["a"], variables=["group"])
    with runtime:
        x = runtime.initialize(selection, {"x": ["group"]})["x"]
        np.testing.assert_array_equal(
            fnp.unique(x), np.unique(frame.to_numpy(dtype="U10"))
        )
        assert runtime.step == 1
        with pytest.raises(ValueError, match="numerical"):
            fnp.min(x)
        with pytest.raises(TypeError):
            fnp.unique(x, return_counts=True)


@pytest.mark.parametrize("kind", [pb.MIN_VALUES, pb.MAX_VALUES, pb.UNION_VALUES])
def test_real_aggregation_kind_and_sequential_consumption(kind):
    service = FetaAggregationService()
    unique = kind == pb.UNION_VALUES
    arrays = (
        [np.array(["a", "λ"]), np.array(["long label", "a"])]
        if unique
        else [np.array([3.0, -4.0]), np.array([-2.0, 5.0])]
    )
    dtype = str(np.result_type(*[a.dtype for a in arrays]))
    service.prepare(
        pb.PrepareRequest(
            context_id="ctx",
            step=1,
            worker_ids=["a", "b"],
            shape=[4] if unique else [2],
            dtype=dtype,
            kind=kind,
        )
    )
    for worker, value in zip(["a", "b"], arrays):
        service.submit(
            pb.SubmitRequest(
                context_id="ctx", step=1, worker_id=worker, value=encode_tensor(value)
            )
        )
    result = service.fetch(pb.StepRequest(context_id="ctx", step=1))
    expected = (
        np.unique(np.concatenate(arrays))
        if unique
        else (
            np.min(arrays, axis=0) if kind == pb.MIN_VALUES else np.max(arrays, axis=0)
        )
    )
    np.testing.assert_array_equal(result, expected)
    service.prepare(
        pb.PrepareRequest(
            context_id="ctx", step=2, worker_ids=["a", "b"], shape=[1], dtype="float64"
        )
    )
    for worker in ["a", "b"]:
        service.submit(
            pb.SubmitRequest(
                context_id="ctx",
                step=2,
                worker_id=worker,
                value=encode_tensor(np.array([2.0])),
            )
        )
    np.testing.assert_array_equal(
        service.fetch(pb.StepRequest(context_id="ctx", step=2)), [4.0]
    )
    service.cleanup("ctx")
    assert not service.sessions


@pytest.mark.parametrize("case", ["count", "width", "dtype"])
def test_invalid_union_contribution_terminates_fetch(case):
    service = FetaAggregationService()
    service.prepare(
        pb.PrepareRequest(
            context_id="ctx",
            step=1,
            worker_ids=["a"],
            shape=[1],
            dtype="U1",
            kind=pb.UNION_VALUES,
        )
    )
    value = (
        np.array(["a", "b"])
        if case == "count"
        else (np.array(["long"]) if case == "width" else np.array([1.0]))
    )
    with pytest.raises(ValueError, match="bounds"):
        service.submit(
            pb.SubmitRequest(
                context_id="ctx", step=1, worker_id="a", value=encode_tensor(value)
            )
        )
    with pytest.raises(ValueError, match="bounds"):
        service.fetch(pb.StepRequest(context_id="ctx", step=1))
    service.cleanup("ctx")


def test_string_transport_checks_width_and_memory():
    with pytest.raises(ValueError, match="dtype"):
        decode_tensor(pb.TensorValue(strings=pb.StringArray(dtype="invalid")), 1024)
    message = pb.TensorValue(strings=pb.StringArray(values=["long"], dtype="U1"))
    with pytest.raises(ValueError, match="width"):
        decode_tensor(message, 1024)
    message.strings.dtype = "U4"
    with pytest.raises(ResourceLimitError):
        decode_tensor(message, 8)
    np.testing.assert_array_equal(decode_tensor(message, 1024), ["long"])
    with pytest.raises(ValueError):
        decode_tensor(pb.TensorValue(), 1024)
    service = FetaAggregationService(
        settings=FetaSettings(max_tensor_bytes=16, max_context_bytes=32)
    )
    with pytest.raises(ResourceLimitError):
        service.prepare(
            pb.PrepareRequest(
                context_id="ctx",
                step=1,
                worker_ids=["a"],
                shape=[8],
                dtype="U2",
                kind=pb.UNION_VALUES,
            )
        )
    assert not service.sessions


def test_numpy_array_functions_remain_local():
    a = np.array([[3.0, -1.0], [2.0, 8.0]])
    np.testing.assert_array_equal(fnp.min(a, axis=0), np.min(a, axis=0))
    np.testing.assert_array_equal(fnp.max(a, axis=1), np.max(a, axis=1))
    np.testing.assert_array_equal(fnp.unique(a), np.unique(a))


def test_unknown_aggregation_kind_has_no_context():
    service = FetaAggregationService()
    with pytest.raises(ValueError, match="kind"):
        service.prepare(
            pb.PrepareRequest(
                context_id="ctx",
                step=1,
                worker_ids=["a"],
                shape=[1],
                dtype="float64",
                kind=99,
            )
        )
    assert not service.sessions


def test_unique_rejects_invalid_worker_descriptor_and_cleans_up():
    frame = pd.DataFrame({"group": ["a", "b", "a"]})
    runtime, workers, aggregation = federation([frame])
    original = workers[0].execute

    def corrupt(request):
        result = original(request)
        result.shape[:] = [100]
        return result

    workers[0].execute = corrupt
    with pytest.raises(ValueError, match="distinct-value shape"):
        with runtime:
            x = runtime.initialize(
                Inputdata(data_model="test:1.0", datasets=["a"], variables=["group"]),
                {"x": ["group"]},
            )["x"]
            fnp.unique(x)
    assert not aggregation.rounds
    assert all(not w.service.store.contexts for w in workers)
