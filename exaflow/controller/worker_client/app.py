import logging
import threading
from typing import Any

import grpc

from exaflow.protos.worker import worker_pb2_grpc
from exaflow.utils import Singleton
from exaflow.worker_communication import BadUserInput
from exaflow.worker_communication import InsufficientDataError

LOGGER = logging.getLogger(__name__)


class WorkerClientConnectionError(Exception):
    def __init__(self, connection_address: str, error_details: str):
        message = f"Connection Error: connection_address={connection_address} error_details={error_details}"
        super().__init__(message)


class WorkerClientTimeoutException(Exception):
    def __init__(self, timeout_type: str, connection_address: str):
        message = f"Timeout Exception: timeout_type={timeout_type} connection_address={connection_address}"
        super().__init__(message)


class WorkerClient:
    def __init__(self, socket_addr: str):
        self._socket_addr = socket_addr
        self._channel = grpc.insecure_channel(socket_addr)
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
        self._lock = threading.Lock()

    def get_client(self, socket_addr: str) -> WorkerClient:
        with self._lock:
            if socket_addr not in self._clients:
                self._clients[socket_addr] = WorkerClient(socket_addr)
            return self._clients[socket_addr]

    def retain_only(self, socket_addrs: set[str]) -> None:
        """
        Drop cached clients for addresses that are no longer in the federation.
        Channels are not closed explicitly, so in-flight calls on a dropped client
        can finish; gRPC releases the channel when the client is garbage collected.
        """
        with self._lock:
            self._clients = {
                addr: client
                for addr, client in self._clients.items()
                if addr in socket_addrs
            }
