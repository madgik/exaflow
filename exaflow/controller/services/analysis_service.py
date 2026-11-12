"""HTTP-independent analysis orchestration."""

from exaflow.controller import config as ctrl_config
from exaflow.controller.services import get_worker_landscape_aggregator
from exaflow.controller.services.algorithm_execution_strategy_factory import (
    get_algorithm_execution_strategy,
)
from exaflow.controller.services.analysis_execution import analysis_slot
from exaflow.controller.services.api.algorithm_spec_dtos import specifications
from exaflow.controller.services.api.analysis_request_dtos import AnalysisRequestDTO
from exaflow.controller.services.api.analysis_request_validator import (
    validate_analysis_request,
)


async def run_analysis(analysis_request_dto: AnalysisRequestDTO) -> str:
    validate_analysis_request(
        analysis_request_dto=analysis_request_dto,
        algorithms_specs=specifications.enabled_algorithms,
        preprocessing_steps_specs=specifications.enabled_preprocessing_steps,
        worker_landscape_aggregator=get_worker_landscape_aggregator(),
        smpc_enabled=ctrl_config.smpc.enabled,
        smpc_optional=ctrl_config.smpc.optional,
    )

    strategy = get_algorithm_execution_strategy(analysis_request_dto)
    async with analysis_slot():
        return await strategy.execute()
