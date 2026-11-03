from __future__ import annotations

import logging

import grpc

import exaflow.protos.aggregation_server.aggregation_server_pb2 as pb2
from exaflow.aggregation_clients import BaseAggregationClient
from exaflow.controller.services.controller_shutdown import register_inflight_rpc

logger = logging.getLogger(__name__)


class ControllerAggregationClient(BaseAggregationClient):
    def _invoke(self, rpc_name: str, request, timeout: float):
        rpc = getattr(self._stub, rpc_name)
        # Generated stubs expose .future(), which shutdown can cancel. In-process
        # test doubles are plain callables and return as soon as they are called.
        future_method = getattr(rpc, "future", None)
        if not callable(future_method):
            return rpc(request, timeout=timeout)
        future = future_method(request, timeout=timeout)
        register_inflight_rpc(future)
        return future.result()

    def configure(self, num_workers: int, timeout: float) -> str:
        response = self._invoke(
            "Configure",
            pb2.ConfigureRequest(
                request_id=self._request_id, num_of_workers=num_workers
            ),
            timeout,
        )
        return response.status

    def cleanup(self, timeout: float) -> str:
        try:
            response = self._invoke(
                "Cleanup",
                pb2.CleanupRequest(request_id=self._request_id),
                timeout,
            )
            return response.status
        except grpc.FutureCancelledError:
            logger.warning(
                "Aggregation cleanup for request_id=%s was cancelled during shutdown",
                self._request_id,
            )
            return pb2.Status.ERROR
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
