from typing import List
from typing import Type

from exaflow import exareme3_preprocessing_step_classes
from exaflow.algorithms.specifications import AlgorithmType
from exaflow.algorithms.specifications import ComponentType
from exaflow.controller.services.api.algorithm_spec_dtos import specifications
from exaflow.controller.services.api.analysis_request_dtos import AnalysisRequestDTO
from exaflow.controller.services.controller_interface import ControllerI
from exaflow.controller.services.exareme3 import (
    get_exareme3_controller as get_exareme3_controller,
)
from exaflow.controller.services.exareme3.strategies import Exareme3Strategy
from exaflow.controller.services.exareme3.strategies import (
    Exareme3WithAggregationServerStrategy,
)
from exaflow.controller.services.flower import (
    get_flower_controller as get_flower_controller,
)
from exaflow.controller.services.flower.strategies import FlowerStrategy
from exaflow.controller.services.strategy_interface import AlgorithmExecutionStrategyI
from exaflow.controller.uid_generator import UIDGenerator


def get_algorithm_execution_strategy(
    analysis_request_dto: AnalysisRequestDTO,
) -> AlgorithmExecutionStrategyI:
    if not analysis_request_dto.request_id:
        analysis_request_dto.request_id = UIDGenerator().get_a_uid()

    algorithm_name = analysis_request_dto.algorithm.name
    algo_type = specifications.get_algorithm_type(algorithm_name)
    components = _get_required_component_types(analysis_request_dto)
    controller = _get_algorithm_controller(algo_type)
    strategy_type = _get_algorithm_strategy_type(algo_type, components)

    return strategy_type(controller, analysis_request_dto)


def _get_algorithm_controller(algo_type: AlgorithmType) -> ControllerI:
    if algo_type in [AlgorithmType.EXAREME3]:
        return get_exareme3_controller()
    elif algo_type == AlgorithmType.FLOWER:
        return get_flower_controller()

    raise NotImplementedError(
        f"Could not get algorithm controller. Unsupported algorithm type: {algo_type}"
    )


def _get_required_component_types(
    analysis_request_dto: AnalysisRequestDTO,
) -> List[ComponentType]:
    algorithm_name = analysis_request_dto.algorithm.name
    components = list(specifications.get_component_types(algorithm_name))
    for preprocessing_step in analysis_request_dto.preprocessing or []:
        preprocessing_step_cls = exareme3_preprocessing_step_classes[
            preprocessing_step.name
        ]
        if preprocessing_step_cls.aggregation_server_required():
            components.append(ComponentType.AGGREGATION_SERVER)
    return list(dict.fromkeys(components))


def _get_algorithm_strategy_type(
    algo_type: AlgorithmType,
    algo_component_types: List[ComponentType],
) -> Type[AlgorithmExecutionStrategyI]:
    if algo_type == AlgorithmType.EXAREME3:
        if ComponentType.AGGREGATION_SERVER in algo_component_types:
            return Exareme3WithAggregationServerStrategy
        return Exareme3Strategy
    elif algo_type == AlgorithmType.FLOWER:
        return FlowerStrategy

    raise NotImplementedError(
        f"Could not get algorithm strategy type. Unsupported algorithm type: {algo_type}"
    )
