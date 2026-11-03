from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Optional

from exaflow.controller.services import WorkerLandscapeAggregator
from exaflow.controller.services.tasks_handler_interface import TasksHandlerI


class ControllerI(ABC):
    """
    The Controller classes are instantiated only once per engine type and they hold
    constant variables used across all algorithm executions.
    """

    worker_landscape_aggregator: WorkerLandscapeAggregator
    task_timeout: int

    def __init__(
        self,
        worker_landscape_aggregator: WorkerLandscapeAggregator,
        task_timeout: int,
    ) -> None:
        self.worker_landscape_aggregator = worker_landscape_aggregator
        self.task_timeout = task_timeout

    @abstractmethod
    def create_worker_tasks_handler(
        self, request_id: str, worker_info
    ) -> TasksHandlerI:
        pass

    def get_local_worker_tasks_handlers(
        self,
        data_model: str,
        datasets: List[str],
        request_id: str,
    ) -> List[TasksHandlerI]:
        workers_info = (
            self.worker_landscape_aggregator.get_workers_info_with_any_of_datasets(
                data_model,
                datasets,
            )
        )

        task_handlers = [
            self.create_worker_tasks_handler(request_id, info) for info in workers_info
        ]

        return task_handlers

    def get_global_worker_tasks_handler(
        self, request_id: str
    ) -> Optional[TasksHandlerI]:
        worker_info = self.worker_landscape_aggregator.get_global_worker()
        if not worker_info:
            return None
        worker_tasks_handler = self.create_worker_tasks_handler(request_id, worker_info)
        return worker_tasks_handler
