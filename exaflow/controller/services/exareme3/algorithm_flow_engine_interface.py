import queue
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from exaflow.algorithms.exareme3.utils.metadata_enums import add_ordered_enums
from exaflow.algorithms.exareme3.utils.registry import get_udf_registry_key
from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.controller import logger as ctrl_logger
from exaflow.controller.services.exareme3.tasks_handler import Exareme3TasksHandler
from exaflow.worker_communication import InsufficientDataError
from exaflow.worker_communication import RunUdfSystemArgs


class Exareme3AlgorithmFlowEngineInterface:
    def __init__(
        self,
        request_id: str,
        context_id: str,
        tasks_handlers: List[Exareme3TasksHandler],
        inputdata: Inputdata,
        metadata: dict,
        preprocessing: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self._logger = ctrl_logger.get_request_logger(request_id=request_id)
        self._context_id = context_id
        self._tasks_handlers = tasks_handlers
        self._preprocessing = preprocessing
        self._inputdata = inputdata
        self._metadata = metadata

    def run_udf(
        self,
        func,
        check_min_rows: bool,
        add_dataset_variable: bool,
        kw_args: dict,
    ) -> List[dict]:
        if not self._tasks_handlers:
            return []

        udf_registry_key = get_udf_registry_key(func)

        system_args = RunUdfSystemArgs(
            inputdata=self._inputdata,
            metadata=add_ordered_enums(self._metadata),
            check_min_rows=check_min_rows,
            add_dataset_variable=add_dataset_variable,
            preprocessing=self._preprocessing,
        )

        done = queue.SimpleQueue()
        calls = []
        try:
            for idx, task_handler in enumerate(self._tasks_handlers):
                udf_call = task_handler.run_udf(
                    udf_registry_key=udf_registry_key,
                    kw_args=kw_args,
                    system_args=system_args,
                )
                udf_call.add_done_callback(lambda _, idx=idx: done.put(idx))
                calls.append(udf_call)
        except Exception:
            for udf_call in calls:
                udf_call.cancel()
            raise

        results: List[Optional[dict]] = [None] * len(self._tasks_handlers)
        for _ in range(len(calls)):
            idx = done.get()
            try:
                results[idx] = calls[idx].result()
            except InsufficientDataError as exc:
                self._logger.warning(
                    "Worker %s returned insufficient data and will be skipped: %s",
                    self._tasks_handlers[idx].worker_id,
                    exc,
                )
                results[idx] = None
            except Exception:
                for udf_call in calls:
                    udf_call.cancel()
                raise

        filtered_results = [res for res in results if res is not None]
        if not filtered_results:
            raise InsufficientDataError(
                "No workers had sufficient data to run the requested algorithm."
            )
        return filtered_results
