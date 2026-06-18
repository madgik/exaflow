from abc import ABC
from abc import abstractmethod
from logging import Logger
from typing import List
from typing import Optional

from exaflow.controller import logger as ctrl_logger
from exaflow.controller.services.api.analysis_request_dtos import AnalysisRequestDTO
from exaflow.controller.services.controller_interface import ControllerI
from exaflow.controller.services.tasks_handler_interface import TasksHandlerI
from exaflow.controller.uid_generator import UIDGenerator


class AlgorithmExecutionStrategyI(ABC):
    """
    The AlgorithmExecutionStrategy holds algorithm execution specific information. It is created and deleted
    along with the algorithm execution lifecycle.
    The Controller class is passed in the strategy init method, and it is used to allow the strategy to use some
    algorithm execution independent variables.
    """

    _controller: ControllerI
    _algorithm_name: str
    _analysis_request_dto: AnalysisRequestDTO
    _request_id: str
    _context_id: str
    _logger: Logger
    _local_worker_tasks_handlers: List[TasksHandlerI]
    _global_worker_tasks_handler: Optional[TasksHandlerI]

    def __init__(
        self,
        controller: ControllerI,
        analysis_request_dto: AnalysisRequestDTO,
    ):
        self._controller = controller
        self._analysis_request_dto = analysis_request_dto
        self._algorithm_name = analysis_request_dto.algorithm.name
        self._request_id = self._analysis_request_dto.request_id
        self._context_id = UIDGenerator().get_a_uid()
        self._logger = ctrl_logger.get_request_logger(self._request_id)
        self._local_worker_tasks_handlers = (
            self._controller.get_local_worker_tasks_handlers(
                self._analysis_request_dto.inputdata.data_model,
                self._analysis_request_dto.inputdata.datasets,
                self._request_id,
            )
        )
        self._global_worker_tasks_handler = (
            self._controller.get_global_worker_tasks_handler(self._request_id)
        )

    @abstractmethod
    async def execute(
        self,
    ) -> str:
        pass
