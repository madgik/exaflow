"""HTTP-independent analysis orchestration.

The route reads the body and calls ``run_analysis``. Callers that tests patch
on the route module are passed in so those patches still apply.
"""

from collections.abc import Callable
from typing import Any

import pydantic

from exaflow.controller import config as ctrl_config
from exaflow.controller.services import get_worker_landscape_aggregator
from exaflow.controller.services.algorithm_execution_strategy_factory import (
    get_algorithm_execution_strategy,
)
from exaflow.controller.services.analysis_execution import analysis_slot
from exaflow.controller.services.api.algorithm_spec_dtos import specifications
from exaflow.controller.services.api.analysis_request_dtos import AnalysisRequestDTO
from exaflow.controller.services.api.analysis_request_validator import BadRequest
from exaflow.controller.services.api.analysis_request_validator import (
    validate_analysis_request,
)


async def run_analysis(
    request_body: Any,
    *,
    validate_analysis_request: Callable = validate_analysis_request,
    get_algorithm_execution_strategy: Callable = get_algorithm_execution_strategy,
    get_worker_landscape_aggregator: Callable = get_worker_landscape_aggregator,
) -> str:
    try:
        analysis_request_dto = AnalysisRequestDTO.model_validate(request_body)
    except pydantic.ValidationError as pydantic_error:
        error_msg = (
            f"Analysis execution request malformed:"
            f"\nrequest received:{request_body}"
            f"\nerror:{pydantic_error}"
        )
        raise BadRequest(error_msg)

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
