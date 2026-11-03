import asyncio
from logging.config import dictConfig

from fastapi import APIRouter
from fastapi import Request
from starlette.responses import JSONResponse

from exaflow.controller.http.loggers import loggers
from exaflow.controller.http.responses import html_response
from exaflow.controller.services import get_worker_landscape_aggregator
from exaflow.controller.services.algorithm_execution_strategy_factory import (
    get_algorithm_execution_strategy,
)
from exaflow.controller.services.analysis_execution import shutdown_analysis_execution
from exaflow.controller.services.analysis_execution import start_analysis_execution
from exaflow.controller.services.analysis_service import (
    run_analysis as execute_analysis,
)
from exaflow.controller.services.api.algorithm_spec_dtos import (
    algorithm_specifications_dtos,
)
from exaflow.controller.services.api.algorithm_spec_dtos import (
    inputdata_specification_dto,
)
from exaflow.controller.services.api.algorithm_spec_dtos import (
    preprocessing_step_specifications_dtos,
)
from exaflow.controller.services.api.analysis_request_validator import (
    validate_analysis_request,
)
from exaflow.controller.services.controller_shutdown import (
    install_shutdown_signal_handlers,
)
from exaflow.controller.services.flower import get_flower_controller
from exaflow.controller.services.flower.flower_io_registry import (
    FLOWER_CALLBACK_MISMATCH_MESSAGE,
)
from exaflow.controller.services.startup import start_background_services
from exaflow.controller.services.startup import stop_background_services

router = APIRouter()


async def startup():
    install_shutdown_signal_handlers()
    configure_loggers()
    start_background_services()
    start_analysis_execution()


async def shutdown():
    try:
        await asyncio.to_thread(stop_background_services)
    finally:
        # A landscape stop that fails must not leave the analysis pool running.
        await asyncio.to_thread(shutdown_analysis_execution, True)


@router.get("/datasets")
async def get_datasets():
    return get_worker_landscape_aggregator().get_all_available_datasets_per_data_model()


@router.get("/datasets_locations")
async def get_datasets_locations():
    return {
        data_model: {
            dataset: worker_id for dataset, worker_id in datasets_location.items()
        }
        for data_model, datasets_location in get_worker_landscape_aggregator()
        .get_datasets_locations()
        .datasets_locations.items()
    }


@router.get("/datasets_variables")
async def get_datasets_variables():
    return {
        data_model: {
            dataset: variables for dataset, variables in datasets_variables.items()
        }
        for data_model, datasets_variables in get_worker_landscape_aggregator()
        .get_datasets_variables()
        .datasets_variables.items()
    }


@router.get("/cdes_metadata")
async def get_cdes_metadata():
    cdes_per_data_model = get_worker_landscape_aggregator().get_cdes_per_data_model()
    return {
        data_model: {
            column: metadata.model_dump() for column, metadata in cdes.values.items()
        }
        for data_model, cdes in cdes_per_data_model.data_models_cdes.items()
    }


@router.get("/data_models_attributes")
async def get_data_models_attributes():
    data_model_attrs = get_worker_landscape_aggregator().get_data_models_attributes()
    return {
        data_model: data_model_metadata.model_dump()
        for data_model, data_model_metadata in data_model_attrs.items()
    }


@router.get("/specifications/inputdata")
async def get_inputdata_specification():
    return html_response(inputdata_specification_dto.model_dump_json())


@router.get("/specifications/preprocessing")
async def get_preprocessing_specifications():
    return html_response(preprocessing_step_specifications_dtos.model_dump_json())


@router.get("/specifications/algorithms")
async def get_algorithm_specifications():
    return html_response(algorithm_specifications_dtos.model_dump_json())


@router.post("/wla")
async def update_wla():
    await asyncio.to_thread(get_worker_landscape_aggregator().update)
    return html_response("")


@router.get("/healthcheck")
async def healthcheck():
    await asyncio.to_thread(get_worker_landscape_aggregator().healthcheck)
    return html_response("")


@router.get("/livez")
async def livez():
    # Liveness only: unlike /healthcheck, it does not depend on worker health.
    return html_response("")


@router.post("/analysis")
async def run_analysis(request: Request):
    # No Pydantic body parameter: FastAPI would answer 422 for a bad body.
    return html_response(
        await execute_analysis(
            await request.json(),
            validate_analysis_request=validate_analysis_request,
            get_algorithm_execution_strategy=get_algorithm_execution_strategy,
            get_worker_landscape_aggregator=get_worker_landscape_aggregator,
        )
    )


@router.get("/flower/input")
async def get_flower_input(request: Request):
    registry = get_flower_controller().flower_execution_info
    if not registry.is_current(request.headers.get("X-Request-Id")):
        return html_response(FLOWER_CALLBACK_MISMATCH_MESSAGE, status_code=409)
    return registry.get_inputdata()


@router.post("/flower/result")
async def set_flower_result(request: Request):
    registry = get_flower_controller().flower_execution_info
    request_id = request.headers.get("X-Request-Id")
    if not registry.is_current(request_id):
        return html_response(FLOWER_CALLBACK_MISMATCH_MESSAGE, status_code=409)
    request_body = await request.json()
    # Re-check inside set_result. The running id can change while the body
    # is read, and nothing awaits between that check and result_ready.set().
    if not await registry.set_result(result=request_body, request_id=request_id):
        return html_response(FLOWER_CALLBACK_MISMATCH_MESSAGE, status_code=409)

    return JSONResponse({"message": "Result set successfully"})


def configure_loggers():
    """
    The loggers should be initialized at app startup, otherwise the configs are overwritten.
    """
    dictConfig(loggers)
