from ipaddress import IPv4Address
from logging import Logger
from typing import List

from google.protobuf import json_format
from google.protobuf import struct_pb2

from exaflow.controller.worker_client.app import WorkerClientFactory
from exaflow.protos.worker import worker_pb2
from exaflow.udf_result_serialization import UDF_RESULT_FORMAT_JSON_BYTES_V1
from exaflow.udf_result_serialization import decode_udf_result
from exaflow.worker_communication import CommonDataElement
from exaflow.worker_communication import CommonDataElements
from exaflow.worker_communication import DataModelAttributes
from exaflow.worker_communication import DatasetInfo
from exaflow.worker_communication import DatasetsInfoPerDataModel
from exaflow.worker_communication import WorkerInfo
from exaflow.worker_communication import WorkerRole


def _proto_worker_role(role_value: int) -> WorkerRole:
    role_name = worker_pb2.WorkerRole.Name(role_value)
    return WorkerRole[role_name]


def _proto_worker_to_model(worker: worker_pb2.WorkerInfo) -> WorkerInfo:
    return WorkerInfo(
        id=worker.id,
        role=_proto_worker_role(worker.role),
        ip=IPv4Address(worker.ip),
        port=worker.port,
    )


def _proto_datasets_to_model(
    response: worker_pb2.ListDatasetsPerDataModelResponse,
) -> DatasetsInfoPerDataModel:
    datasets_info = {}
    for data_model in response.datasets_info:
        datasets_info[data_model.data_model] = [
            DatasetInfo(
                code=dataset.code,
                label=dataset.label,
                variables=list(dataset.variables or []),
            )
            for dataset in data_model.datasets
        ]
    return DatasetsInfoPerDataModel(datasets_info_per_data_model=datasets_info)


def _proto_common_data_elements(
    response: worker_pb2.GetDataModelCdesResponse,
) -> CommonDataElements:
    values = {}
    for code, element in response.cdes.values.items():
        enumerations_map = dict(element.enumerations)
        if element.enumerations_order:
            ordered_items = []
            for enum_code in element.enumerations_order:
                if enum_code in enumerations_map:
                    ordered_items.append((enum_code, enumerations_map.pop(enum_code)))
            ordered_items.extend(enumerations_map.items())
            enumerations = dict(ordered_items)
        else:
            enumerations = enumerations_map
        values[code] = CommonDataElement(
            code=element.code,
            label=element.label,
            sql_type=element.sql_type,
            is_categorical=element.is_categorical,
            enumerations=enumerations or None,
            min=element.min.value if element.HasField("min") else None,
            max=element.max.value if element.HasField("max") else None,
        )
    return CommonDataElements(values=values)


def _proto_attributes_to_model(
    response: worker_pb2.GetDataModelAttributesResponse,
) -> DataModelAttributes:
    properties = (
        json_format.MessageToDict(
            response.attributes.properties, preserving_proto_field_name=True
        )
        if response.attributes.properties
        else {}
    )
    return DataModelAttributes(
        tags=list(response.attributes.tags or []), properties=properties
    )


def _dict_to_struct(data: dict) -> struct_pb2.Struct:
    message = struct_pb2.Struct()
    json_format.ParseDict(data, message, ignore_unknown_fields=True)
    return message


class WorkerTaskResult:
    """Deprecated placeholder kept for backward compatibility."""


class WorkerTasksHandler:
    def __init__(self, worker_queue_addr: str, logger: Logger):
        self._worker_queue_addr = worker_queue_addr
        self._logger = logger

    def _client(self):
        return WorkerClientFactory().get_client(self._worker_queue_addr)

    def get_worker_info(self, request_id: str, timeout: int) -> WorkerInfo:
        response = self._client().call(
            "GetWorkerInfo",
            worker_pb2.GetWorkerInfoRequest(request_id=request_id),
            timeout,
        )
        return _proto_worker_to_model(response.worker)

    def get_worker_datasets_per_data_model(
        self, request_id: str, timeout: int
    ) -> DatasetsInfoPerDataModel:
        response = self._client().call(
            "ListDatasetsPerDataModel",
            worker_pb2.ListDatasetsPerDataModelRequest(request_id=request_id),
            timeout,
        )
        return _proto_datasets_to_model(response)

    def get_data_model_cdes(
        self, request_id: str, data_model: str, timeout: int
    ) -> CommonDataElements:
        response = self._client().call(
            "GetDataModelCdes",
            worker_pb2.GetDataModelRequest(
                request_id=request_id, data_model=data_model
            ),
            timeout,
        )
        return _proto_common_data_elements(response)

    def get_data_model_attributes(
        self, request_id: str, data_model: str, timeout: int
    ) -> DataModelAttributes:
        response = self._client().call(
            "GetDataModelAttributes",
            worker_pb2.GetDataModelRequest(
                request_id=request_id, data_model=data_model
            ),
            timeout,
        )
        return _proto_attributes_to_model(response)

    def healthcheck(self, request_id: str, check_db: bool, timeout: int) -> None:
        self._client().call(
            "Healthcheck",
            worker_pb2.HealthcheckRequest(request_id=request_id, check_db=check_db),
            timeout,
        )

    def start_flower_client(
        self,
        request_id: str,
        algorithm_folder_path: str,
        server_address: str,
        data_model: str,
        datasets: List[str],
        execution_timeout: int,
        timeout: int,
    ) -> int:
        response = self._client().call(
            "StartFlowerClient",
            worker_pb2.StartFlowerClientRequest(
                request_id=request_id,
                algorithm_folder_path=algorithm_folder_path,
                server_address=server_address,
                data_model=data_model,
                datasets=datasets,
                execution_timeout=execution_timeout,
            ),
            timeout,
        )
        return response.pid

    def start_flower_server(
        self,
        request_id: str,
        algorithm_folder_path: str,
        number_of_clients: int,
        server_address: str,
        data_model: str,
        datasets: List[str],
        timeout: int,
    ) -> int:
        response = self._client().call(
            "StartFlowerServer",
            worker_pb2.StartFlowerServerRequest(
                request_id=request_id,
                algorithm_folder_path=algorithm_folder_path,
                number_of_clients=number_of_clients,
                server_address=server_address,
                data_model=data_model,
                datasets=datasets,
            ),
            timeout,
        )
        return response.pid

    def stop_flower_server(
        self, request_id: str, pid: int, algorithm_name: str, timeout: int
    ) -> None:
        self._client().call(
            "StopFlowerServer",
            worker_pb2.StopFlowerProcessRequest(
                request_id=request_id, pid=pid, algorithm_name=algorithm_name
            ),
            timeout,
        )

    def stop_flower_client(
        self, request_id: str, pid: int, algorithm_name: str, timeout: int
    ) -> None:
        self._client().call(
            "StopFlowerClient",
            worker_pb2.StopFlowerProcessRequest(
                request_id=request_id, pid=pid, algorithm_name=algorithm_name
            ),
            timeout,
        )

    def garbage_collect(self, request_id: str, timeout: int) -> None:
        self._client().call(
            "GarbageCollect",
            worker_pb2.GarbageCollectRequest(request_id=request_id),
            timeout,
        )

    def run_udf(
        self,
        request_id: str,
        udf_registry_key: str,
        kw_args: dict,
        system_args: dict,
        timeout: int,
    ):
        response = self._client().call(
            "RunUdf",
            worker_pb2.RunUdfRequest(
                request_id=request_id,
                udf_registry_key=udf_registry_key,
                kw_args=_dict_to_struct(kw_args),
                system_args=_dict_to_struct(system_args),
            ),
            timeout,
        )
        if response.result_format != UDF_RESULT_FORMAT_JSON_BYTES_V1:
            raise ValueError(
                f"Unsupported RunUdfResponse result format: {response.result_format!r}"
            )
        return decode_udf_result(response.result)
