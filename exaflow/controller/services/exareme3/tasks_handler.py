from exaflow.controller import logger as ctrl_logger
from exaflow.controller.services.tasks_handler_interface import TasksHandlerI
from exaflow.controller.worker_client.tasks_handler import WorkerTasksHandler
from exaflow.worker_communication import RunUdfSystemArgs


class Exareme3TasksHandler(TasksHandlerI):
    def __init__(
        self,
        request_id: str,
        worker_id: str,
        worker_queue_addr: str,
        tasks_timeout: int,
    ):
        self._request_id = request_id
        self._worker_id = worker_id
        self._worker_queue_addr = worker_queue_addr
        self._tasks_timeout = tasks_timeout
        self._logger = ctrl_logger.get_request_logger(request_id=request_id)
        self._worker_tasks_handler = WorkerTasksHandler(
            self._worker_queue_addr, self._logger
        )

    @property
    def worker_id(self) -> str:
        return self._worker_id

    @property
    def tasks_timeout(self) -> int:
        return self._tasks_timeout

    def run_udf(
        self, udf_registry_key: str, kw_args: dict, system_args: RunUdfSystemArgs
    ):
        return self._worker_tasks_handler.run_udf(
            request_id=self._request_id,
            udf_registry_key=udf_registry_key,
            kw_args=kw_args,
            system_args=system_args.model_dump(),
            timeout=self._tasks_timeout,
        )
