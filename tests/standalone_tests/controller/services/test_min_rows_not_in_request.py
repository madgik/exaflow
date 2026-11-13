"""The controller no longer tells workers whether to check the minimum row count.

The worker decides from its own UDF registry, so the flag is gone from the
algorithm base class, the flow engine and the payload sent to workers.
"""

import inspect
from unittest.mock import Mock

from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.controller.services.exareme3.algorithm_flow_engine_interface import (
    Exareme3AlgorithmFlowEngineInterface,
)


def _udf(data):
    return {}


def _inputdata():
    return Inputdata(data_model="dm:1", datasets=["ds"], variables=["x"])


class _FinishedCall:
    def __init__(self, result):
        self._result = result

    def add_done_callback(self, fn):
        fn(self)

    def cancel(self):
        return False

    def result(self):
        return self._result


class _RecordingHandler:
    worker_id = "worker-0"

    def __init__(self):
        self.system_args = None
        self.kw_args = None

    def run_udf(self, udf_registry_key, kw_args, system_args):
        self.kw_args = kw_args
        self.system_args = system_args
        return _FinishedCall({"ok": True})


class _RecordingEngine:
    def __init__(self):
        self.call = None

    def run_udf(self, func, add_dataset_variable, kw_args):
        self.call = (func, add_dataset_variable, kw_args)
        return [{"value": 1}]


class _DatasetAwareAlgorithm(Algorithm):
    @property
    def add_dataset_variable(self) -> bool:
        return True

    def run(self):
        return None


def test_flow_engine_run_udf_takes_no_minimum_row_argument():
    parameters = inspect.signature(
        Exareme3AlgorithmFlowEngineInterface.run_udf
    ).parameters

    assert list(parameters) == ["self", "func", "add_dataset_variable", "kw_args"]


def test_flow_engine_sends_no_minimum_row_flag_to_workers():
    handler = _RecordingHandler()
    engine = Exareme3AlgorithmFlowEngineInterface(
        request_id="req-min-rows",
        context_id="ctx",
        tasks_handlers=[handler],
        inputdata=_inputdata(),
        metadata={},
    )

    results = engine.run_udf(func=_udf, add_dataset_variable=True, kw_args={"a": 1})

    assert results == [{"ok": True}]
    assert handler.kw_args == {"a": 1}
    assert handler.system_args.add_dataset_variable is True
    assert "check_min_rows" not in handler.system_args.model_dump()


def test_algorithm_base_class_has_no_check_min_rows_property():
    assert not hasattr(Algorithm, "check_min_rows")


def test_run_local_udf_passes_function_dataset_flag_and_arguments():
    engine = _RecordingEngine()
    algorithm = _DatasetAwareAlgorithm(
        engine=engine,
        logger=Mock(),
        inputdata=_inputdata(),
        x=["x"],
        parameters={},
    )

    results = algorithm.run_local_udf(_udf, kw_args={"a": 1})

    assert results == [{"value": 1}]
    assert engine.call == (_udf, True, {"a": 1})
