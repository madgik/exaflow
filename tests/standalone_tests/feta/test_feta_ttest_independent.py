import numpy as np
import pandas as pd
import pytest
from scipy import stats

from exaflow.algorithms.federated.statistics.ttest_independent import (
    FederatedTTestIndependent,
)
from exaflow.algorithms.feta import federated_numpy as fnp
from exaflow.algorithms.feta.feta_ttest_independent import FetaTTestIndependent
from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.worker_communication import BadUserInput
from tests.standalone_tests.feta.helpers import federation
from tests.standalone_tests.feta.test_feta_pearson_correlation import (
    IdentityAggregation,
)


def selection():
    return Inputdata(
        data_model="test:1.0", datasets=["a"], variables=["group", "value"]
    )


def assert_reference(result, frame, parameters):
    reference = FederatedTTestIndependent(IdentityAggregation()).compute(
        data=frame,
        group_var="group",
        value_var="value",
        group_a=parameters["groupA"],
        group_b=parameters["groupB"],
        alpha=parameters.get("alpha", 0.05),
        alternative=parameters.get("alt_hypothesis", "two-sided"),
    )
    reference["p"] = reference.pop("p_value")
    for key, expected in reference.items():
        if isinstance(expected, str):
            assert getattr(result, key) == expected
        else:
            np.testing.assert_allclose(
                getattr(result, key), expected, rtol=1e-10, atol=1e-12, equal_nan=True
            )


@pytest.mark.parametrize("sizes", [[45], [15, 19, 11], [0, 25, 20]])
@pytest.mark.parametrize("minimum_rows", [0, 12])
@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
@pytest.mark.parametrize("alpha", [0.05, 0.1])
def test_independent_parity(sizes, minimum_rows, alternative, alpha):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        {
            "group": np.resize(["control", "treated", "other"], sum(sizes)),
            "value": rng.normal(size=sum(sizes)),
        }
    )
    edges = np.cumsum([0, *sizes])
    parts = [frame.iloc[a:b] for a, b in zip(edges[:-1], edges[1:])]
    eligible = pd.concat([p for p in parts if len(p) >= minimum_rows])
    parameters = dict(
        groupA="control", groupB="treated", alpha=alpha, alt_hypothesis=alternative
    )
    runtime, workers, aggregation = federation(parts, minimum_rows=minimum_rows)
    with runtime:
        tensors = runtime.initialize(selection(), {"x": ["group"], "y": ["value"]})
        result = FetaTTestIndependent(parameters=parameters).run(**tensors)
        assert_reference(result, eligible, parameters)
        reference = stats.ttest_ind(
            eligible.value[eligible.group == "control"],
            eligible.value[eligible.group == "treated"],
            alternative=alternative,
        )
        np.testing.assert_allclose(
            [result.t_stat, result.p],
            [reference.statistic, reference.pvalue],
            atol=1e-12,
        )
        assert runtime.step == 6
    assert not aggregation.rounds
    assert all(not w.service.store.contexts for w in workers)


@pytest.mark.parametrize(
    "groups, values, error",
    [
        (["a", "a"], [1.0, 2.0], "Group B"),
        (["b", "b"], [1.0, 2.0], "Group A"),
        (["a", "b"], [1.0, 2.0], "Not enough"),
        (["a", "b", "b"], [1.0, 2.0, 4.0], None),
        (["a", "a", "b", "b"], [1.0, 1.0, 1.0, 1.0], None),
    ],
)
def test_small_groups_and_constants(groups, values, error):
    frame = pd.DataFrame({"group": groups, "value": values})
    parameters = dict(groupA="a", groupB="b")
    runtime, _, _ = federation([frame])
    with runtime:
        tensors = runtime.initialize(selection(), {"x": ["group"], "y": ["value"]})
        if error:
            with pytest.raises(BadUserInput, match=error):
                FetaTTestIndependent(parameters=parameters).run(**tensors)
        else:
            assert_reference(
                FetaTTestIndependent(parameters=parameters).run(**tensors),
                frame,
                parameters,
            )


def test_string_masks_alignment_and_empty_local_groups():
    frame = pd.DataFrame(
        {"group": ["α", "α", "long label", "long label"], "value": [1.0, 3.0, 5.0, 8.0]}
    )
    runtime, _, _ = federation([frame.iloc[:2], frame.iloc[2:]])
    with runtime:
        tensors = runtime.initialize(selection(), {"x": ["group"], "y": ["value"]})
        x, y = tensors["x"], tensors["y"]
        mask = fnp.equal(x[:, 0], "α")
        a, b = y[mask], y[mask]
        assert a.shape == (2, 1)
        assert fnp.sum(a + b) == 8
        with pytest.raises(ValueError, match="unaligned"):
            a + y[x[:, 0] == "long label"]
        with pytest.raises(ValueError, match="numerical"):
            x + "foo"
        with pytest.raises(TypeError, match="ambiguous"):
            bool(mask)
        with pytest.raises(ValueError, match="one-dimensional"):
            y[x == "α"]
        with pytest.raises(TypeError, match="indexing"):
            y[0]
        parameters = dict(groupA="α", groupB="long label")
        assert_reference(
            FetaTTestIndependent(parameters=parameters).run(**tensors),
            frame,
            parameters,
        )


def test_protocol_literals_and_filter_validation():
    from exaflow.feta.operations import LocalOperation
    from exaflow.feta.operations import LocalTensorRef
    from exaflow.feta.operations import OperationType
    from exaflow.protos.feta import feta_pb2 as pb

    operation = LocalOperation(OperationType.EQUAL, (LocalTensorRef("a"), "λ"))
    assert LocalOperation.from_proto(operation.to_proto(), 1024) == operation
    boolean = LocalOperation(OperationType.EQUAL, (LocalTensorRef("a"), True))
    assert LocalOperation.from_proto(boolean.to_proto(), 1024) == boolean
    array_operation = LocalOperation(
        OperationType.EQUAL, (np.array([True, False]), True)
    )
    decoded = LocalOperation.from_proto(array_operation.to_proto(), 1024)
    assert decoded.inputs[0].dtype == np.dtype("bool")
    np.testing.assert_array_equal(decoded.inputs[0], [True, False])
    from exaflow.feta.lifecycle import ResourceLimitError

    with pytest.raises(ResourceLimitError, match="string literal exceeds"):
        LocalOperation.from_proto(operation.to_proto(), 1)
    malformed = operation.to_proto()
    malformed.selection_layout = "unexpected"
    frame = pd.DataFrame({"group": ["a", "b"], "value": [1.0, 2.0]})
    runtime, workers, _ = federation([frame])
    with runtime:
        tensors = runtime.initialize(selection(), {"x": ["group"], "y": ["value"]})
        malformed.inputs[0].tensor_id = tensors["x"].descriptors["worker-0"].tensor_id
        with pytest.raises(ValueError, match="Only FILTER"):
            workers[0].service.execute(
                pb.ExecuteRequest(context_id=runtime.context_id, operation=malformed)
            )
        mask = tensors["x"][:, 0] == "a"
        assert fnp.sum(mask) == 1
        assert fnp.sum(mask == True) == 1
        assert fnp.sum(mask + np.array([True])) == 2


@pytest.mark.parametrize("kind", ["mixed", "oversized", "missing"])
def test_string_initialization_errors_cleanup(kind):
    from exaflow.feta.lifecycle import ResourceLimitError
    from exaflow.feta.settings import FetaSettings

    frame = pd.DataFrame({"group": ["a", "b"], "value": [1.0, 2.0]})
    settings = FetaSettings(max_tensor_bytes=32, max_context_bytes=64)
    columns = {"x": ["group"], "y": ["value"]}
    if kind == "mixed":
        columns = {"x": ["group", "value"]}
    elif kind == "oversized":
        frame["group"] = ["a" * 30, "b"]
    else:
        frame.loc[0, "group"] = None
    runtime, workers, _ = federation([frame], settings=settings)
    with pytest.raises((ValueError, ResourceLimitError)):
        with runtime:
            runtime.initialize(selection(), columns)
    assert all(not w.service.store.contexts for w in workers)
