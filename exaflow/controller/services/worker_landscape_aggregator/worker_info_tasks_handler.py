from exaflow.controller import logger as ctrl_logger
from exaflow.controller.worker_client.tasks_handler import WorkerTasksHandler
from exaflow.worker_communication import CommonDataElements
from exaflow.worker_communication import DataModelAttributes
from exaflow.worker_communication import DatasetsInfoPerDataModel
from exaflow.worker_communication import WorkerInfo


class WorkerInfoTasksHandler:
    def __init__(self, worker_queue_addr: str, tasks_timeout: int, request_id: str):
        self._worker_queue_addr = worker_queue_addr
        self._tasks_timeout = tasks_timeout
        self._request_id = request_id
        self._logger = ctrl_logger.get_request_logger(request_id=request_id)
        self._worker_tasks_handler = WorkerTasksHandler(
            self._worker_queue_addr, self._logger
        )

    def get_worker_info_task(self) -> WorkerInfo:
        return self._worker_tasks_handler.get_worker_info(
            request_id=self._request_id, timeout=self._tasks_timeout
        )

    def get_worker_datasets_per_data_model_task(self) -> DatasetsInfoPerDataModel:
        return self._worker_tasks_handler.get_worker_datasets_per_data_model(
            request_id=self._request_id, timeout=self._tasks_timeout
        )

    def get_data_model_cdes_task(self, data_model: str) -> CommonDataElements:
        return self._worker_tasks_handler.get_data_model_cdes(
            request_id=self._request_id,
            data_model=data_model,
            timeout=self._tasks_timeout,
        )

    def get_data_model_attributes_task(self, data_model: str) -> DataModelAttributes:
        return self._worker_tasks_handler.get_data_model_attributes(
            request_id=self._request_id,
            data_model=data_model,
            timeout=self._tasks_timeout,
        )

    def get_healthcheck_task(self, check_db: bool):
        return self._worker_tasks_handler.healthcheck(
            request_id=self._request_id,
            check_db=check_db,
            timeout=self._tasks_timeout,
        )
