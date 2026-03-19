import argparse
import json
import logging
from concurrent import futures
from typing import Optional

import grpc
from google.protobuf import json_format
from google.protobuf import struct_pb2
from google.protobuf import wrappers_pb2
from grpc_health.v1 import health
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc

from exaflow.protos.worker import worker_pb2
from exaflow.protos.worker import worker_pb2_grpc
from exaflow.worker import config as worker_config
from exaflow.worker.exareme3.udf import udf_service
from exaflow.worker.utils import duck_db_csv_loader
from exaflow.worker.worker_info import worker_info_service
from exaflow.worker_communication import BadUserInput
from exaflow.worker_communication import CommonDataElement
from exaflow.worker_communication import CommonDataElements
from exaflow.worker_communication import DataModelAttributes
from exaflow.worker_communication import DatasetsInfoPerDataModel
from exaflow.worker_communication import InsufficientDataError
from exaflow.worker_communication import RunUdfSystemArgs
from exaflow.worker_communication import WorkerInfo

LOGGER = logging.getLogger("WorkerGrpcServer")
WORKER_HEALTH_SERVICE_NAME = "worker"


class FlowerNotAvailableError(RuntimeError):
    """Raised when Flower dependencies are not installed on the worker."""


def _get_flower_services():
    try:
        from exaflow.worker.flower.cleanup import cleanup_service
        from exaflow.worker.flower.starter import starter_service
    except ModuleNotFoundError as exc:
        raise FlowerNotAvailableError(
            "Flower support is not available on this worker. "
            "Rebuild the worker image with the 'flower' dependency group "
            "or disable FLOWER_ENABLED and avoid Flower algorithms."
        ) from exc
    return cleanup_service, starter_service


def _worker_role_to_proto(role) -> int:
    return worker_pb2.WorkerRole.Value(role.name)


def _worker_info_to_proto(info: WorkerInfo) -> worker_pb2.WorkerInfo:
    return worker_pb2.WorkerInfo(
        id=info.id,
        role=_worker_role_to_proto(info.role),
        ip=str(info.ip),
        port=info.port,
    )


def _datasets_info_to_proto(
    datasets_info: DatasetsInfoPerDataModel,
) -> list[worker_pb2.DatasetsInfoPerDataModel]:
    proto_items = []
    for data_model, datasets in datasets_info.datasets_info_per_data_model.items():
        proto_items.append(
            worker_pb2.DatasetsInfoPerDataModel(
                data_model=data_model,
                datasets=[
                    worker_pb2.DatasetInfo(
                        code=dataset.code,
                        label=dataset.label,
                        variables=list(dataset.variables or []),
                    )
                    for dataset in datasets
                ],
            )
        )
    return proto_items


def _common_data_element_to_proto(
    element: CommonDataElement,
) -> worker_pb2.CommonDataElement:
    enumerations = element.enumerations if element.enumerations is not None else {}
    enumerations_order = list(enumerations.keys())
    proto_element = worker_pb2.CommonDataElement(
        code=element.code,
        label=element.label,
        sql_type=element.sql_type,
        is_categorical=element.is_categorical,
        enumerations=enumerations,
        enumerations_order=enumerations_order,
    )

    if element.min is not None:
        proto_element.min.CopyFrom(wrappers_pb2.DoubleValue(value=float(element.min)))
    if element.max is not None:
        proto_element.max.CopyFrom(wrappers_pb2.DoubleValue(value=float(element.max)))
    return proto_element


def _common_data_elements_to_proto(
    cdes: CommonDataElements,
) -> worker_pb2.CommonDataElements:
    return worker_pb2.CommonDataElements(
        values={
            code: _common_data_element_to_proto(element)
            for code, element in cdes.values.items()
        }
    )


def _dict_to_struct(data: dict) -> struct_pb2.Struct:
    struct = struct_pb2.Struct()
    json_format.ParseDict(data, struct, ignore_unknown_fields=True)
    return struct


def _dict_to_value(data) -> struct_pb2.Value:
    proto_value = struct_pb2.Value()
    json_format.Parse(json.dumps(data), proto_value)
    return proto_value


def _data_model_attributes_to_proto(
    attributes: DataModelAttributes,
) -> worker_pb2.DataModelAttributes:
    props = attributes.properties or {}
    return worker_pb2.DataModelAttributes(
        tags=list(attributes.tags or []), properties=_dict_to_struct(props)
    )


def _struct_to_dict(struct_message: struct_pb2.Struct) -> dict:
    return json_format.MessageToDict(struct_message, preserving_proto_field_name=True)


class WorkerService(worker_pb2_grpc.WorkerServiceServicer):
    def __init__(self, *, health_servicer: Optional[health.HealthServicer] = None):
        self._health_servicer = health_servicer
        self._set_health_status(health_pb2.HealthCheckResponse.NOT_SERVING)
        try:
            duck_db_csv_loader.load_all_csvs_from_data_folder(request_id="startup")
            LOGGER.info("Data folder loaded successfully on startup.")
            self._set_health_status(health_pb2.HealthCheckResponse.SERVING)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to load data folder on startup", exc_info=exc)
            self._set_health_status(health_pb2.HealthCheckResponse.NOT_SERVING)

    def _set_health_status(self, status: int):
        if self._health_servicer is None:
            return
        self._health_servicer.set(WORKER_HEALTH_SERVICE_NAME, status)

    def _handle_exception(self, context: grpc.ServicerContext, exc: Exception):
        if isinstance(exc, BadUserInput):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        elif isinstance(exc, InsufficientDataError):
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        elif isinstance(exc, FlowerNotAvailableError):
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        else:
            LOGGER.exception("Worker gRPC call failed", exc_info=exc)
            context.abort(grpc.StatusCode.INTERNAL, str(exc))

    def GetWorkerInfo(self, request, context):
        LOGGER.info(
            "GetWorkerInfo called with request_id=%s",
            request.request_id,
        )
        try:
            info = worker_info_service.get_worker_info(request.request_id)
            return worker_pb2.GetWorkerInfoResponse(worker=_worker_info_to_proto(info))
        except Exception as exc:  # noqa: BLE001
            self._handle_exception(context, exc)

    def ListDatasetsPerDataModel(self, request, context):
        try:
            datasets = worker_info_service.get_worker_datasets_per_data_model(
                request.request_id
            )
            return worker_pb2.ListDatasetsPerDataModelResponse(
                datasets_info=_datasets_info_to_proto(datasets)
            )
        except Exception as exc:  # noqa: BLE001
            self._handle_exception(context, exc)

    def GetDataModelCdes(self, request, context):
        try:
            cdes = worker_info_service.get_data_model_cdes(
                request.request_id, request.data_model
            )
            return worker_pb2.GetDataModelCdesResponse(
                cdes=_common_data_elements_to_proto(cdes)
            )
        except Exception as exc:  # noqa: BLE001
            self._handle_exception(context, exc)

    def GetDataModelAttributes(self, request, context):
        try:
            attributes = worker_info_service.get_data_model_attributes(
                request.request_id, request.data_model
            )
            return worker_pb2.GetDataModelAttributesResponse(
                attributes=_data_model_attributes_to_proto(attributes)
            )
        except Exception as exc:  # noqa: BLE001
            self._handle_exception(context, exc)

    def Healthcheck(self, request, context):
        try:
            worker_info_service.healthcheck(request.request_id, request.check_db)
            return worker_pb2.HealthcheckResponse(ok=True)
        except Exception as exc:  # noqa: BLE001
            self._handle_exception(context, exc)

    def StartFlowerClient(self, request, context):
        try:
            _, starter_service = _get_flower_services()
            pid = starter_service.start_flower_client(
                request_id=request.request_id,
                algorithm_folder_path=request.algorithm_folder_path,
                server_address=request.server_address,
                data_model=request.data_model,
                datasets=list(request.datasets),
                execution_timeout=request.execution_timeout,
            )
            return worker_pb2.StartFlowerClientResponse(pid=pid)
        except Exception as exc:  # noqa: BLE001
            self._handle_exception(context, exc)

    def StartFlowerServer(self, request, context):
        try:
            _, starter_service = _get_flower_services()
            pid = starter_service.start_flower_server(
                request_id=request.request_id,
                algorithm_folder_path=request.algorithm_folder_path,
                number_of_clients=request.number_of_clients,
                server_address=request.server_address,
                data_model=request.data_model,
                datasets=list(request.datasets),
            )
            return worker_pb2.StartFlowerServerResponse(pid=pid)
        except Exception as exc:  # noqa: BLE001
            self._handle_exception(context, exc)

    def StopFlowerClient(self, request, context):
        try:
            cleanup_service, _ = _get_flower_services()
            cleanup_service.stop_flower_process(
                request_id=request.request_id,
                pid=request.pid,
                algorithm_name=request.algorithm_name,
            )
            return worker_pb2.StopFlowerProcessResponse(stopped=True)
        except Exception as exc:  # noqa: BLE001
            self._handle_exception(context, exc)

    def StopFlowerServer(self, request, context):
        try:
            cleanup_service, _ = _get_flower_services()
            cleanup_service.stop_flower_process(
                request_id=request.request_id,
                pid=request.pid,
                algorithm_name=request.algorithm_name,
            )
            return worker_pb2.StopFlowerProcessResponse(stopped=True)
        except Exception as exc:  # noqa: BLE001
            self._handle_exception(context, exc)

    def GarbageCollect(self, request, context):
        try:
            cleanup_service, _ = _get_flower_services()
            cleanup_service.garbage_collect(request.request_id)
            return worker_pb2.GarbageCollectResponse(ok=True)
        except Exception as exc:  # noqa: BLE001
            self._handle_exception(context, exc)

    def RunUdf(self, request, context):
        try:
            kw_args = _struct_to_dict(request.kw_args)
            system_args = RunUdfSystemArgs.model_validate(
                _struct_to_dict(request.system_args)
            )
            result = udf_service.run_udf(
                request_id=request.request_id,
                udf_registry_key=request.udf_registry_key,
                kw_args=kw_args,
                system_args=system_args,
            )
            return worker_pb2.RunUdfResponse(result=_dict_to_value(result))
        except Exception as exc:  # noqa: BLE001
            self._handle_exception(context, exc)


def serve() -> None:
    logging.basicConfig(
        level=worker_config.framework_log_level,
        format=(
            "%(asctime)s - %(levelname)s - %(module)s.%(funcName)s(%(lineno)d) - "
            f"[{worker_config.federation}] - [exaflow-{worker_config.role.lower()}] - "
            f"[{worker_config.identifier}] - [BACKGROUND] - %(message)s"
        ),
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    worker_service = WorkerService(health_servicer=health_servicer)
    worker_pb2_grpc.add_WorkerServiceServicer_to_server(worker_service, server)

    listen_addr = f"{worker_config.grpc.ip}:{worker_config.grpc.port}"
    server.add_insecure_port(listen_addr)
    LOGGER.info("Worker gRPC server listening on %s", listen_addr)
    LOGGER.info("Server starting on %s", listen_addr)
    LOGGER.info("Registered services: exaflow.worker.api.WorkerService")
    LOGGER.info(
        "Service full name from descriptor: %s",
        worker_pb2.DESCRIPTOR.services_by_name["WorkerService"].full_name,
    )

    # Add TestService
    def test_method(request, context):
        LOGGER.info("TestService.Test called")
        return worker_pb2.HealthcheckResponse(ok=True)

    rpc_method_handlers = {
        "Test": grpc.unary_unary_rpc_method_handler(
            test_method,
            request_deserializer=worker_pb2.HealthcheckRequest.FromString,
            response_serializer=worker_pb2.HealthcheckResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "TestService", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))

    server.start()
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        LOGGER.info("Worker gRPC server shutting down...")
        server.stop(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worker-id",
        help="Optional worker identifier used for process discovery",
        default=None,
    )
    args = parser.parse_args()
    if args.worker_id and args.worker_id != worker_config.identifier:
        LOGGER.warning(
            "CLI worker-id '%s' does not match configured identifier '%s'",
            args.worker_id,
            worker_config.identifier,
        )
    serve()
