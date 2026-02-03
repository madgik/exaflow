import logging
import socket
from typing import Sequence

import grpc
import numpy as np
from numpy.typing import NDArray

import exaflow.protos.aggregation_server.aggregation_server_pb2 as pb2
import exaflow.protos.aggregation_server.aggregation_server_pb2_grpc as pb2_grpc

from .constants import AggregationType
from .serialization import bytes_to_ndarray
from .serialization import bytes_to_values
from .serialization import ndarray_to_bytes
from .serialization import values_to_bytes

logger = logging.getLogger(__name__)


ArrayInput = NDArray | Sequence[object]

DEFAULT_AGGREGATION_PORT = "50051"


class BaseAggregationClient:
    def __init__(self, request_id: str, aggregator_dns: str | None = None):
        self._request_id = request_id
        self._step_counter = 1

        target = aggregator_dns or f"172.17.0.1:{DEFAULT_AGGREGATION_PORT}"
        # If a DNS name is provided, ensure a port is present and resolve it
        if aggregator_dns:
            host, port = (
                aggregator_dns.rsplit(":", 1)
                if ":" in aggregator_dns
                else (aggregator_dns, DEFAULT_AGGREGATION_PORT)
            )
            target = f"{host}:{port}"
            try:
                resolved = socket.gethostbyname(host)
                target = f"{resolved}:{port}"
            except Exception as exc:
                logger.warning(
                    "Failed to resolve aggregator DNS %s: %s", aggregator_dns, exc
                )

        # Increase gRPC message size limits to support large Arrow tensors (e.g., ~80 MiB)
        GRPC_MAX_MESSAGE_SIZE = 100 * 1024 * 1024  # 100 MiB
        self._channel = grpc.insecure_channel(
            target,
            options=[
                ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_SIZE),
                ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_SIZE),
            ],
        )
        self._stub = pb2_grpc.AggregationServerStub(self._channel)

    def _next_step(self) -> int:
        step = self._step_counter
        self._step_counter += 1
        return step

    def _aggregate_request(
        self,
        aggregation_type: AggregationType,
        flat_values: ArrayInput,
        *,
        step: int | None = None,
    ) -> NDArray[np.floating]:
        results = self._aggregate_batch_request(
            [(aggregation_type, flat_values)], step=step
        )
        return results[0]

    def _aggregate_batch_request(
        self, ops: list[tuple[AggregationType, ArrayInput]], *, step: int | None = None
    ) -> list[NDArray[np.floating]]:
        operations = []
        for op, vals in ops:
            if op == AggregationType.UNION:
                tensor = values_to_bytes(vals)
            else:
                array = np.asarray(vals, dtype=np.float64)
                tensor = ndarray_to_bytes(array)
            operations.append(
                pb2.Operation(
                    aggregation_type=op.value,
                    tensor=tensor,
                )
            )
        req = pb2.AggregateRequest(
            request_id=self._request_id,
            step=step or self._next_step(),
            operations=operations,
        )
        resp = self._stub.Aggregate(req)
        if resp.tensors:
            decoded = []
            for (op, _), tensor in zip(ops, resp.tensors):
                if op == AggregationType.UNION:
                    decoded.append(bytes_to_values(tensor))
                else:
                    decoded.append(bytes_to_ndarray(tensor))
            return decoded

        results = resp.results
        offsets = resp.offsets
        reconstructed = []
        for i in range(len(offsets) - 1):
            start, end = offsets[i], offsets[i + 1]
            reconstructed.append(np.asarray(results[start:end], dtype=np.float64))
        return reconstructed

    def close(self) -> None:
        logger.debug("[CHANNEL] Closing gRPC channel.")
        self._channel.close()
