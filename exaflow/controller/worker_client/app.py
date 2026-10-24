import logging
import re
from typing import Any

import grpc

from exaflow.controller import config as ctrl_config
from exaflow.grpc_options import grpc_max_message_size
from exaflow.grpc_options import grpc_message_options
from exaflow.protos.worker import worker_pb2_grpc
from exaflow.utils import Singleton
from exaflow.worker_communication import BadUserInput
from exaflow.worker_communication import InsufficientDataError

LOGGER = logging.getLogger(__name__)
_GRPC_SIZE_RE = re.compile(r"\((\d+)\s+vs\.?\s+(\d+)\)")


def _format_mib(num_bytes: int) -> str:
    mib = num_bytes / (1024 * 1024)
    if abs(mib - round(mib)) < 0.05:
        return f"{round(mib):.0f} MiB"
    return f"{mib:.1f} MiB"


def _message_too_large_text(error_details: str, max_message_bytes: int) -> str:
    details = error_details or ""
    details_lower = details.lower()
    # The controller is the gRPC client: what it sends, or the worker receives,
    # is the request; what the worker sends, or the controller receives, is the
    # result.
    if "sent message larger" in details_lower:
        kind = "request" if "client:" in details_lower else "result"
    elif "received message larger" in details_lower:
        kind = "result" if "client:" in details_lower else "request"
    else:
        kind = "result"
    match = _GRPC_SIZE_RE.search(details)
    if match:
        sent = int(match.group(1))
        limit = int(match.group(2))
        return (
            f"{kind} too large ({sent} bytes / {_format_mib(sent)} > "
            f"{limit} bytes / {_format_mib(limit)}); reduce variables/levels"
        )
    return (
        f"{kind} too large (>{max_message_bytes} bytes / "
        f"{_format_mib(max_message_bytes)}); reduce variables/levels"
    )


class WorkerClientConnectionError(Exception):
    def __init__(self, connection_address: str, error_details: str):
        message = f"Connection Error: connection_address={connection_address} error_details={error_details}"
        super().__init__(message)


class WorkerMessageTooLargeError(Exception):
    def __init__(
        self, connection_address: str, error_details: str, max_message_bytes: int
    ):
        self.connection_address = connection_address
        self.error_details = error_details
        self.max_message_bytes = max_message_bytes
        self.message = _message_too_large_text(error_details, max_message_bytes)
        super().__init__(self.message)


class WorkerClientTimeoutException(Exception):
    def __init__(self, timeout_type: str, connection_address: str):
        message = f"Timeout Exception: timeout_type={timeout_type} connection_address={connection_address}"
        super().__init__(message)


class WorkerClient:
    def __init__(self, socket_addr: str):
        self._socket_addr = socket_addr
        self._max_message_bytes = grpc_max_message_size(
            ctrl_config.grpc.max_message_mb
        )
        self._channel = grpc.insecure_channel(
            socket_addr,
            options=grpc_message_options(ctrl_config.grpc.max_message_mb),
        )
        self._stub = worker_pb2_grpc.WorkerServiceStub(self._channel)

    def close(self) -> None:
        self._channel.close()

    def _handle_rpc_error(self, exc: grpc.RpcError) -> None:
        status_code = exc.code()
        details = exc.details() or ""
        if status_code == grpc.StatusCode.DEADLINE_EXCEEDED:
            raise WorkerClientTimeoutException(
                timeout_type=str(status_code), connection_address=self._socket_addr
            ) from exc

        if status_code == grpc.StatusCode.INVALID_ARGUMENT:
            raise BadUserInput(details)

        if status_code == grpc.StatusCode.FAILED_PRECONDITION:
            raise InsufficientDataError(details)

        if status_code == grpc.StatusCode.RESOURCE_EXHAUSTED:
            raise WorkerMessageTooLargeError(
                connection_address=self._socket_addr,
                error_details=details or str(status_code),
                max_message_bytes=self._max_message_bytes,
            ) from exc

        raise WorkerClientConnectionError(
            connection_address=self._socket_addr,
            error_details=details or str(status_code),
        ) from exc

    def call(self, rpc_name: str, request: Any, timeout: int):
        rpc = getattr(self._stub, rpc_name)
        try:
            return rpc(request, timeout=timeout)
        except grpc.RpcError as exc:  # noqa: BLE001
            LOGGER.debug(
                "gRPC call %s to %s failed with status %s",
                rpc_name,
                self._socket_addr,
                exc.code(),
            )
            self._handle_rpc_error(exc)


class WorkerClientFactory(metaclass=Singleton):
    def __init__(self):
        self._clients: dict[str, WorkerClient] = {}

    def get_client(self, socket_addr: str) -> WorkerClient:
        if socket_addr in self._clients:
            return self._clients[socket_addr]

        self._clients[socket_addr] = WorkerClient(socket_addr)
        return self._clients[socket_addr]
