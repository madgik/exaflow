import sys
import types

from exaflow.algorithms.specifications import ComponentType

dns_module = types.ModuleType("dns")
dns_resolver_module = types.ModuleType("dns.resolver")
dns_module.resolver = dns_resolver_module
sys.modules.setdefault("dns", dns_module)
sys.modules.setdefault("dns.resolver", dns_resolver_module)

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


class _NoAggregationPreprocessing:
    @classmethod
    def aggregation_server_required(cls):
        return False


class _AggregationPreprocessing:
    @classmethod
    def aggregation_server_required(cls):
        return True


class _Specifications:
    def get_component_types(self, algo_name):
        return []


def _request(preprocessing=None):
    return AnalysisRequestDTO(
        inputdata=AnalysisInputDataDTO(
            data_model="dementia",
            datasets=["ppmi0"],
            variables=["x"],
        ),
        preprocessing=preprocessing,
        algorithm=AnalysisAlgorithmDTO(name="sample_algo", y=["x"]),
    )


def test_preprocessing_aggregation_requirement_contributes_component(monkeypatch):
    monkeypatch.setattr(factory, "specifications", _Specifications())
    monkeypatch.setattr(
        factory,
        "exareme3_preprocessing_step_classes",
        {
            "local": _NoAggregationPreprocessing,
            "global": _AggregationPreprocessing,
        },
    )

    components = factory._get_required_component_types(
        _request(
            preprocessing=[
                AnalysisPreprocessingStepDTO(name="local", parameters={}),
                AnalysisPreprocessingStepDTO(name="global", parameters={}),
            ]
        )
    )

    assert ComponentType.AGGREGATION_SERVER in components


def test_preprocessing_aggregation_component_selects_aggregation_strategy():
    strategy_type = factory._get_algorithm_strategy_type(
        algo_type=factory.AlgorithmType.EXAREME3,
        algo_component_types=[ComponentType.AGGREGATION_SERVER],
    )

    assert strategy_type is Exareme3WithAggregationServerStrategy


def test_no_aggregation_component_selects_plain_exareme3_strategy():
    strategy_type = factory._get_algorithm_strategy_type(
        algo_type=factory.AlgorithmType.EXAREME3,
        algo_component_types=[],
    )

    assert strategy_type is Exareme3Strategy
