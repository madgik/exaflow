import numpy as np

import exaflow.protos.aggregation_server.aggregation_server_pb2 as pb2
from exaflow.aggregation_clients import AggregationType
from exaflow.aggregation_clients import BaseAggregationClient
from exaflow.algorithms.federated.utils.agg_client import ArrayInput


class Exareme3UDFAggregationClient(BaseAggregationClient):
    """Aggregation client injected into UDFs as `agg_client`.

    It satisfies the `AggregationClient` protocol that the federated algorithm
    code is written against.
    """

    def sum(self, values: ArrayInput) -> np.ndarray:
        return self.aggregate(AggregationType.SUM, values)

    def min(self, values: ArrayInput) -> np.ndarray:
        return self.aggregate(AggregationType.MIN, values)

    def max(self, values: ArrayInput) -> np.ndarray:
        return self.aggregate(AggregationType.MAX, values)

    def union(self, values: list[object]) -> np.ndarray:
        return self.aggregate(AggregationType.UNION, values)

    def aggregate(
        self, aggregation_type: AggregationType, values: ArrayInput
    ) -> np.ndarray:
        original = np.asarray(values)
        flat = original.ravel()

        aggregated = self._aggregate_request(aggregation_type, flat)

        if original.shape and aggregated.size == original.size:
            return aggregated.reshape(original.shape)

        return aggregated

    def aggregate_batch(
        self, ops: list[tuple[AggregationType, ArrayInput]]
    ) -> list[np.ndarray]:
        originals = [np.asarray(vals) for _, vals in ops]
        agg_types = [agg for agg, _ in ops]
        flat_ops = [
            (agg_type, arr.ravel()) for agg_type, arr in zip(agg_types, originals)
        ]
        aggregated_lists = self._aggregate_batch_request(flat_ops)
        results = []
        for arr, agg_np in zip(originals, aggregated_lists):
            if arr.shape and agg_np.size == arr.size:
                agg_np = agg_np.reshape(arr.shape)
            results.append(agg_np)
        return results

    def unregister(self) -> tuple[str, int]:
        response = self._stub.Unregister(
            pb2.UnregisterRequest(request_id=self._request_id)
        )
        return response.status, response.remaining_workers
