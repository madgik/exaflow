from types import SimpleNamespace

from exaflow.algorithms.specifications import AlgorithmType
from exaflow.algorithms.specifications import ComponentType
from exaflow.controller.services import algorithm_execution_strategy_factory as factory
from exaflow.controller.services.api.analysis_request_dtos import AnalysisAlgorithmDTO
from exaflow.controller.services.api.analysis_request_dtos import AnalysisInputDataDTO
from exaflow.controller.services.api.analysis_request_dtos import (
    AnalysisPreprocessingStepDTO,
)
from exaflow.controller.services.api.analysis_request_dtos import AnalysisRequestDTO
from exaflow.controller.services.exareme3.strategies import Exareme3Strategy
from exaflow.controller.services.exareme3.strategies import (
    Exareme3WithAggregationServerStrategy,
)


class _FakeController:
    def get_local_worker_tasks_handlers(self, data_model, datasets, request_id):
        return []

    def get_global_worker_tasks_handler(self, request_id):
        return None


class _FakeSpecifications:
    def __init__(self):
        self.algorithm_components = []
        self.enabled_preprocessing_steps = {
            "agg_step": SimpleNamespace(
                components=[ComponentType.AGGREGATION_SERVER],
            )
        }

    def get_algorithm_type(self, algorithm_name):
        return AlgorithmType.EXAREME3

    def get_component_types(self, algorithm_name):
        return self.algorithm_components


def _request(*, preprocessing=None):
    return AnalysisRequestDTO(
        inputdata=AnalysisInputDataDTO(
            data_model="dementia:0.1",
            datasets=["dataset_a"],
            variables=["x", "y"],
        ),
        preprocessing=preprocessing,
        algorithm=AnalysisAlgorithmDTO(
            name="plain_algorithm",
            x=["x"],
            y=["y"],
            parameters={},
        ),
        flags={},
    )


def test_strategy_factory_does_not_mutate_algorithm_components(monkeypatch):
    fake_specifications = _FakeSpecifications()
    monkeypatch.setattr(factory, "specifications", fake_specifications)
    monkeypatch.setattr(
        factory,
        "_get_algorithm_controller",
        lambda algo_type: _FakeController(),
    )

    with_aggregation_preprocessing = factory.get_algorithm_execution_strategy(
        _request(
            preprocessing=[
                AnalysisPreprocessingStepDTO(name="agg_step", parameters={})
            ],
        )
    )
    without_preprocessing = factory.get_algorithm_execution_strategy(_request())

    assert isinstance(
        with_aggregation_preprocessing,
        Exareme3WithAggregationServerStrategy,
    )
    assert isinstance(without_preprocessing, Exareme3Strategy)
    assert not isinstance(
        without_preprocessing,
        Exareme3WithAggregationServerStrategy,
    )
    assert fake_specifications.algorithm_components == []
