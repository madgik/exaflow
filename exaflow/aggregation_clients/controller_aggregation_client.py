from __future__ import annotations

import grpc

import exaflow.protos.aggregation_server.aggregation_server_pb2 as pb2
from exaflow.aggregation_clients import BaseAggregationClient


class ControllerAggregationClient(BaseAggregationClient):
    def configure(self, num_workers: int) -> str:
        response = self._stub.Configure(
            pb2.ConfigureRequest(
                request_id=self._request_id, num_of_workers=num_workers
            )
        )
        return response.status

    def cleanup(self) -> str:
        try:
            response = self._stub.Cleanup(
                pb2.CleanupRequest(request_id=self._request_id)
            )
            return response.status
        except grpc._channel._InactiveRpcError:
            # aggregation_server already shut down remotely – not an error
            return pb2.Status.ERROR

    def __enter__(self) -> ControllerAggregationClient:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
