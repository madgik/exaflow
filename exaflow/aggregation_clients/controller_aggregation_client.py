from __future__ import annotations

import logging

import grpc

import exaflow.protos.aggregation_server.aggregation_server_pb2 as pb2
from exaflow.aggregation_clients import BaseAggregationClient
from exaflow.controller.services.controller_shutdown import register_inflight_rpc
from exaflow.controller.services.errors import ControllerShuttingDownError

logger = logging.getLogger(__name__)


class ControllerAggregationClient(BaseAggregationClient):
    def configure(self, num_workers: int, timeout: float) -> str:
        future = self._stub.Configure.future(
            pb2.ConfigureRequest(
                request_id=self._request_id, num_of_workers=num_workers
            ),
            timeout=timeout,
        )
        # Shutdown cancels a Configure that waits on a hung server.
        register_inflight_rpc(future)
        try:
            return future.result().status
        except grpc.FutureCancelledError:
            raise ControllerShuttingDownError() from None

    def cleanup(self, timeout: float) -> str:
        try:
            response = self._stub.Cleanup(
                pb2.CleanupRequest(request_id=self._request_id), timeout=timeout
            )
            return response.status
        except grpc.RpcError as exc:
            # Cleanup is what releases workers still waiting on this request, so
            # a failure must be visible.
            logger.warning(
                "Aggregation cleanup for request_id=%s failed: %s %s",
                self._request_id,
                exc.code(),
                exc.details(),
            )
            return pb2.Status.ERROR

    def __enter__(self) -> ControllerAggregationClient:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
