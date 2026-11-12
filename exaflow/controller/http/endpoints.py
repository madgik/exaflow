import asyncio
from logging.config import dictConfig
from typing import Annotated
from typing import Any

from fastapi import APIRouter
from fastapi import Body

from exaflow.controller.http.loggers import loggers
from exaflow.controller.http.responses import html_response
from exaflow.controller.services import analysis_service
from exaflow.controller.services import get_worker_landscape_aggregator
from exaflow.controller.services.analysis_execution import shutdown_analysis_execution
from exaflow.controller.services.analysis_execution import start_analysis_execution
from exaflow.controller.services.api.algorithm_spec_dtos import (
    algorithm_specifications_dtos,
)
from exaflow.controller.services.api.algorithm_spec_dtos import (
    inputdata_specification_dto,
)
from exaflow.controller.services.api.algorithm_spec_dtos import (
    preprocessing_step_specifications_dtos,
)
from exaflow.controller.services.api.analysis_request_dtos import AnalysisRequestDTO
from exaflow.controller.services.controller_shutdown import SHUTDOWN_RPC_CANCEL_TIMEOUT
from exaflow.controller.services.controller_shutdown import begin_controller_shutdown
from exaflow.controller.services.flower import get_flower_controller
from exaflow.controller.services.startup import start_background_services
from exaflow.controller.services.startup import stop_background_services

router = APIRouter()


async def startup():
    configure_loggers()
    start_background_services()
    start_analysis_execution()


async def shutdown():
    # Uvicorn has drained the requests that could finish and cancelled the
    # others. Their analysis threads keep running until their RPCs are
    # cancelled, so cancel them here.
    begin_controller_shutdown()
    try:
        await asyncio.to_thread(stop_background_services)
    finally:
        # A landscape stop that fails must not leave the analysis pool running.
        await asyncio.to_thread(
            shutdown_analysis_execution, SHUTDOWN_RPC_CANCEL_TIMEOUT
        )


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
async def run_analysis(analysis_request: AnalysisRequestDTO):
    return html_response(await analysis_service.run_analysis(analysis_request))


@router.get("/flower/input")
async def get_flower_input():
    return get_flower_controller().flower_execution_info.get_inputdata()


@router.post("/flower/result")
async def set_flower_result(result: Annotated[Any, Body()]):
    await get_flower_controller().flower_execution_info.set_result(result=result)
    return {"message": "Result set successfully"}


def configure_loggers():
    """
    The loggers should be initialized at app startup, otherwise the configs are overwritten.
    """
    dictConfig(loggers)
