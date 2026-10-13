import numpy as np
import pytest

from exaflow.algorithms.feta import FedTensor
from tests.standalone_tests.feta.helpers import federation
from tests.standalone_tests.feta.test_feta_pearson_correlation import data
from tests.standalone_tests.feta.test_feta_pearson_correlation import inputs


@pytest.mark.parametrize(
    "columns",
    [
        {"y": ["y0"]},
        {"x": ["x0"]},
        {"x": ["x1", "x0"], "y": ["y0"]},
        {"x": ["x0"], "y": ["x0"]},
    ],
)
def test_only_selected_groups_are_loaded_and_stored(columns):
    frame = data(39)
    runtime, workers, _ = federation([frame.iloc[:17], frame.iloc[17:]])
    with runtime:
        tensors = runtime.initialize(inputs(frame), columns)
        assert set(tensors) == set(columns)
        for worker in workers:
            context = worker.service.store.contexts[runtime.context_id]
            assert len(context.values) == len({tuple(c) for c in columns.values()})
        for name, selected in columns.items():
            assert isinstance(tensors[name], FedTensor)
            assert tensors[name].shape == (39, len(selected))
            np.testing.assert_allclose(
                tensors[name].sum(axis=0), frame[selected].sum().to_numpy()
            )
    assert all(not worker.service.store.contexts for worker in workers)


def test_missing_values_in_unselected_group_are_not_loaded():
    frame = data(20)
    frame["x0"] = np.nan
    runtime, _, _ = federation([frame])
    with runtime:
        tensors = runtime.initialize(inputs(frame), {"y": ["y0"]})
        np.testing.assert_allclose(tensors["y"].sum(), frame.y0.sum())


@pytest.mark.parametrize("columns", [{}, {"z": ["x0"]}, {"y": []}])
def test_invalid_group_selections_fail_before_loading(columns):
    frame = data(20)
    runtime, workers, _ = federation([frame])
    with runtime:
        with pytest.raises(ValueError, match="selected x or y"):
            runtime.initialize(inputs(frame), columns)
        assert all(not worker.service.store.contexts for worker in workers)


@pytest.mark.parametrize("fault", ["missing", "extra", "shape", "alignment"])
def test_mismatched_worker_group_response_fails_and_cleans_context(fault):
    frame = data(20)
    runtime, workers, _ = federation([frame])
    original = workers[0].initialize

    def initialize(request):
        response = original(request)
        if fault == "missing":
            response.ClearField("y")
        elif fault == "extra":
            response.x.CopyFrom(response.y)
        elif fault == "shape":
            response.y.shape[1] = 2
        else:
            response.x.shape[0] += 1
        return response

    workers[0].initialize = initialize
    columns = {"y": ["y0"]}
    if fault == "alignment":
        columns["x"] = ["x0"]
    with pytest.raises(ValueError, match="groups|metadata|unaligned"):
        with runtime:
            runtime.initialize(inputs(frame), columns)
    assert not workers[0].service.store.contexts
