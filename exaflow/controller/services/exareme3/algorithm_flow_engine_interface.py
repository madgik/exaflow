from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
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
        preprocessing: dict = None,
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
        drop_na: bool,
        check_min_rows: bool,
        add_dataset_variable: bool,
        kw_args: dict,
    ) -> List[dict]:
        if not self._tasks_handlers:
            return []

        udf_registry_key = get_udf_registry_key(func)

        if "metadata" in kw_args:
            kw_args["metadata"] = add_ordered_enums(kw_args["metadata"])
        system_metadata = add_ordered_enums(self._metadata)

        system_args = RunUdfSystemArgs(
            inputdata=self._inputdata,
            metadata=system_metadata,
            drop_na=drop_na,
            check_min_rows=check_min_rows,
            add_dataset_variable=add_dataset_variable,
            preprocessing=self._preprocessing,
        )

        executor = ThreadPoolExecutor(max_workers=len(self._tasks_handlers))
        future_to_index = {}
        pending = set()
        results: List[Optional[dict]] = [None] * len(self._tasks_handlers)
        try:
            future_to_index = {
                executor.submit(
                    task_handler.run_udf,
                    udf_registry_key=udf_registry_key,
                    kw_args=kw_args,
                    system_args=system_args,
                ): idx
                for idx, task_handler in enumerate(self._tasks_handlers)
            }
            pending = set(future_to_index)
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except InsufficientDataError as exc:
                    self._logger.warning(
                        "Worker %s returned insufficient data and will be skipped: %s",
                        self._tasks_handlers[idx].worker_id,
                        exc,
                    )
                    results[idx] = None
                pending.discard(future)
        except Exception:
            for future in future_to_index:
                future.cancel()
            raise
        finally:
            executor.shutdown(wait=not pending, cancel_futures=bool(pending))
        filtered_results = [res for res in results if res is not None]
        if not filtered_results:
            raise InsufficientDataError(
                "No workers had sufficient data to run the requested algorithm."
            )
        return filtered_results
