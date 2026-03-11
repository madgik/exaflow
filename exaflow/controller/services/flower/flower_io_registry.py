import asyncio
from enum import Enum
from enum import unique
from typing import Any
from typing import Dict
from typing import Optional


@unique
class Status(str, Enum):
    SUCCESS = "success"
    RUNNING = "running"
    FAILURE = "failure"


class Result:
    def __init__(self, content: Dict[str, Any], status: Status):
        self.content = content
        self.status = status

    def __repr__(self):
        return f"Result(status={self.status}, content={self.content})"


class FlowerIORegistry:
    def __init__(self, timeout, logger):
        self._inputdata: Optional[dict] = None
        self._parameters: Optional[dict] = None
        self._run_env: Optional[dict] = None
        self._events: Optional[list] = None
        self._request_id: Optional[str] = None
        self._algorithm_name: Optional[str] = None
        self._result: Optional[Result] = None
        self.result_ready: Optional[asyncio.Event] = None
        self._logger = logger
        self._reset_sync()
        self._timeout = timeout

    def _reset_sync(self):
        """Synchronously resets the algorithm execution info to initial state."""
        self._inputdata = {}
        self._parameters = {}
        self._run_env = {}
        self._events = []
        self._request_id = None
        self._algorithm_name = None
        self._result = Result(content={}, status=Status.RUNNING)
        self.result_ready = asyncio.Event()
        self._logger.debug("Algorithm reset: input data cleared, status set to RUNNING")

    async def reset(self):
        """Asynchronously resets the algorithm execution info to initial state."""
        self._reset_sync()
        self._logger.debug("Asynchronous reset complete")

    async def set_result(self, result: Dict[str, Any]):
        """Sets the execution result and updates the status based on the presence of an error."""
        status = Status.FAILURE if "error" in result else Status.SUCCESS
        self._result = Result(content=result, status=status)
        self._logger.debug(f"Result set with status: {status}, content: {result}")
        self.result_ready.set()

    async def get_result(self) -> Dict[str, Any]:
        await self.result_ready.wait()
        self._logger.debug(f"Result retrieved: {self._result}")
        return self._result.content

    async def get_result_with_timeout(self) -> Dict[str, Any]:
        try:
            await asyncio.wait_for(self.get_result(), self._timeout)
            self._logger.debug(f"Result with timeout: {self._result}")
            return self._result.content
        except asyncio.TimeoutError:
            error = f"Failed to get result: operation timed out after {self._timeout} seconds"
            self._logger.error(error)
            self._result = Result(content={"error": error}, status=Status.FAILURE)
            raise TimeoutError(error)

    def get_status(self) -> Status:
        """Returns the current status of the execution."""
        status = self._result.status if self._result else Status.RUNNING
        self._logger.debug(f"Status retrieved: {status}")
        return status

    def set_inputdata(self, inputdata: dict):
        """Sets new input data for the algorithm and resets status and result."""
        self._inputdata = inputdata
        self._result = Result(content={}, status=Status.RUNNING)
        self.result_ready.clear()
        self._logger.debug(f"Input data updated: {inputdata}, status reset to RUNNING")

    def get_inputdata(self) -> dict:
        """Returns the current input data."""
        self._logger.debug(f"Input data retrieved: {self._inputdata}")
        return self._inputdata

    def set_parameters(self, parameters: dict):
        """Sets validated algorithm parameters for Flower processes."""
        self._parameters = parameters or {}
        self._logger.debug("Flower parameters updated.")

    def get_parameters(self) -> dict:
        """Returns current algorithm parameters."""
        self._logger.debug("Flower parameters retrieved.")
        return self._parameters

    def set_run_env(self, run_env: dict):
        """Sets computed runtime env overrides for Flower processes."""
        self._run_env = run_env or {}
        self._logger.debug("Flower run env updated.")

    def get_run_env(self) -> dict:
        """Returns current runtime env overrides."""
        self._logger.debug("Flower run env retrieved.")
        return self._run_env

    async def add_event(self, event: Dict[str, Any]):
        """Adds a runtime event payload (e.g. round_event)."""
        self._events.append(event)
        self._logger.debug("Flower event appended. total_events=%s", len(self._events))

    def get_events(self) -> list:
        """Returns runtime events for the active execution."""
        return list(self._events or [])

    def set_execution_context(self, request_id: str, algorithm_name: str):
        """Sets active execution context for scoped Flower access."""
        self._request_id = request_id
        self._algorithm_name = algorithm_name
        self._logger.debug(
            "Flower execution context set. request_id=%s algorithm=%s",
            request_id,
            algorithm_name,
        )

    def validate_execution_scope(self, request_id: Optional[str]) -> bool:
        """Validates that the caller request id matches the active Flower execution."""
        if request_id is None:
            # Backward-compatible access for existing Flower algorithms.
            return True
        if not self._request_id:
            return False
        return request_id == self._request_id
