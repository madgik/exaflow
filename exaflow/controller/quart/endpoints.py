from logging.config import dictConfig

import pydantic
from quart import Blueprint
from quart import abort
from quart import jsonify
from quart import request

from exaflow.controller import config as ctrl_config
from exaflow.controller.quart.loggers import loggers
from exaflow.controller.services import get_worker_landscape_aggregator
from exaflow.controller.services.algorithm_execution_strategy_factory import (
    get_algorithm_execution_strategy,
)
from exaflow.controller.services.api.algorithm_request_dtos import AlgorithmRequestDTO
from exaflow.controller.services.api.algorithm_request_validator import BadRequest
from exaflow.controller.services.api.algorithm_request_validator import (
    validate_algorithm_request,
)
from exaflow.controller.services.api.algorithm_spec_dtos import (
    algorithm_specifications_dtos,
)
from exaflow.controller.services.api.algorithm_spec_dtos import specifications
from exaflow.controller.services.flower import get_flower_controller
from exaflow.controller.services.startup import start_background_services

algorithms = Blueprint("algorithms_endpoint", __name__)


@algorithms.before_app_serving
async def startup():
    configure_loggers()
    start_background_services()


@algorithms.route("/datasets", methods=["GET"])
async def get_datasets() -> dict:
    return get_worker_landscape_aggregator().get_all_available_datasets_per_data_model()


@algorithms.route("/datasets_locations", methods=["GET"])
async def get_datasets_locations() -> dict:
    return {
        data_model: {
            dataset: worker_id for dataset, worker_id in datasets_location.items()
        }
        for data_model, datasets_location in get_worker_landscape_aggregator()
        .get_datasets_locations()
        .datasets_locations.items()
    }


@algorithms.route("/datasets_variables", methods=["GET"])
async def get_datasets_variables() -> dict:
    return {
        data_model: {
            dataset: variables for dataset, variables in datasets_variables.items()
        }
        for data_model, datasets_variables in get_worker_landscape_aggregator()
        .get_datasets_variables()
        .datasets_variables.items()
    }


@algorithms.route("/cdes_metadata", methods=["GET"])
async def get_cdes_metadata() -> dict:
    cdes_per_data_model = get_worker_landscape_aggregator().get_cdes_per_data_model()
    return {
        data_model: {
            column: metadata.dict() for column, metadata in cdes.values.items()
        }
        for data_model, cdes in cdes_per_data_model.data_models_cdes.items()
    }


@algorithms.route("/data_models_attributes", methods=["GET"])
async def get_data_models_attributes() -> dict:
    data_model_attrs = get_worker_landscape_aggregator().get_data_models_attributes()
    return {
        data_model: data_model_metadata.dict()
        for data_model, data_model_metadata in data_model_attrs.items()
    }


@algorithms.route("/algorithms", methods=["GET"])
async def get_algorithms() -> str:
    return algorithm_specifications_dtos.json()


@algorithms.route("/wla", methods=["POST"])
async def update_wla() -> str:
    get_worker_landscape_aggregator().update()
    return ""


@algorithms.route("/healthcheck", methods=["GET"])
async def healthcheck() -> str:
    get_worker_landscape_aggregator().healthcheck()
    return ""


@algorithms.route("/algorithms/<algorithm_name>", methods=["POST"])
async def run_algorithm(algorithm_name: str) -> str:
    request_body = await request.json
    try:
        algorithm_request_dto = AlgorithmRequestDTO.parse_obj(request_body)
    except pydantic.error_wrappers.ValidationError as pydantic_error:
        error_msg = (
            f"Algorithm execution request malformed:"
            f"\nrequest received:{request_body}"
            f"\nerror:{pydantic_error}"
        )
        raise BadRequest(error_msg)

    validate_algorithm_request(
        algorithm_name=algorithm_name,
        algorithm_request_dto=algorithm_request_dto,
        algorithms_specs=specifications.enabled_algorithms,
        transformers_specs=specifications.enabled_transformers,
        worker_landscape_aggregator=get_worker_landscape_aggregator(),
        smpc_enabled=ctrl_config.smpc.enabled,
        smpc_optional=ctrl_config.smpc.optional,
    )

    strategy = get_algorithm_execution_strategy(algorithm_name, algorithm_request_dto)
    return await strategy.execute()


@algorithms.route("/flower/input", methods=["GET"])
async def get_flower_input() -> dict:
    request_id = request.args.get("request_id")
    if not get_flower_controller().flower_execution_info.validate_execution_scope(
        request_id
    ):
        abort(400, description="Flower request scope mismatch.")
    return get_flower_controller().flower_execution_info.get_inputdata()


@algorithms.route("/flower/parameters", methods=["GET"])
async def get_flower_parameters() -> dict:
    request_id = request.args.get("request_id")
    if not get_flower_controller().flower_execution_info.validate_execution_scope(
        request_id
    ):
        abort(400, description="Flower request scope mismatch.")
    return get_flower_controller().flower_execution_info.get_parameters()


@algorithms.route("/flower/run_env", methods=["GET"])
async def get_flower_run_env() -> dict:
    request_id = request.args.get("request_id")
    if not get_flower_controller().flower_execution_info.validate_execution_scope(
        request_id
    ):
        abort(400, description="Flower request scope mismatch.")
    return get_flower_controller().flower_execution_info.get_run_env()


@algorithms.route("/flower/events", methods=["GET"])
async def get_flower_events() -> dict:
    request_id = request.args.get("request_id")
    if not get_flower_controller().flower_execution_info.validate_execution_scope(
        request_id
    ):
        abort(400, description="Flower request scope mismatch.")
    return {"events": get_flower_controller().flower_execution_info.get_events()}


@algorithms.route("/flower/event", methods=["POST"])
async def add_flower_event():
    request_id = request.args.get("request_id")
    if not get_flower_controller().flower_execution_info.validate_execution_scope(
        request_id
    ):
        abort(400, description="Flower request scope mismatch.")
    request_body = await request.json
    await get_flower_controller().flower_execution_info.add_event(event=request_body)
    return jsonify({"message": "Event added successfully"}), 200


@algorithms.route("/flower/result", methods=["POST"])
async def set_flower_result():
    request_body = await request.json
    await get_flower_controller().flower_execution_info.set_result(result=request_body)

    return jsonify({"message": "Result set successfully"}), 200


def configure_loggers():
    """
    The loggers should be initialized at app startup, otherwise the configs are overwritten.
    """
    dictConfig(loggers)
