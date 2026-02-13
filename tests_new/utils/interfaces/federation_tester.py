import sys
import grpc
import os
import time
from concurrent import futures as grpc_futures


from exaflow.aggregation_server.server import AggregationServer
from exaflow.aggregation_server import config
from exaflow.protos.aggregation_server import aggregation_server_pb2
from exaflow.protos.aggregation_server import aggregation_server_pb2_grpc
from grpc_health.v1 import health
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc
from abc import ABC, abstractmethod

from library.utils.aggregators.exaflow_aggregation_client import ExaflowAggregationClient
from tests_new.utils.interfaces.partitioned_table import PartitionedPandasTable
from concurrent.futures import ThreadPoolExecutor, as_completed

AGGREGATOR_DNS="localhost:50051"

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

class FederationTester(ABC):

    request_id=0

    def __init__(self, num_of_workers,request_id,*, dataset: PartitionedPandasTable,**kwargs):
        self.num_of_workers = num_of_workers
        self.request_id = request_id
        self.dataset = dataset
        self.kwargs=kwargs

    def _run_client(self, client_id, local_dataset,**kwargs):

        print(f"[{client_id}] Started.")

        # Initialize implementation of AggregationClientInterface
        client = ExaflowAggregationClient(request_id=self.request_id, aggregator_dns=AGGREGATOR_DNS)
        try:
            results = self.federated_computation(client,local_dataset,**kwargs)
            client.close()
            print(f"[{client_id}] [OK] All aggregations completed.")
            return results
        except Exception as e:
            print(f"[{client_id}] [ERROR] Error during aggregation: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _configure_server(self):
        channel = grpc.insecure_channel(AGGREGATOR_DNS)
        stub = aggregation_server_pb2_grpc.AggregationServerStub(channel)

        print(f"[Controller] Configuring server for '{self.request_id}' with {self.num_of_workers} workers...")
        try:
            stub.Configure(
                aggregation_server_pb2.ConfigureRequest(
                    request_id=self.request_id,
                    num_of_workers=self.num_of_workers
                )
            )
            print("[Controller] Server configured successfully.")
        except grpc.RpcError as e:
            print(f"[Controller] Failed to configure server: {e}")

    @classmethod
    def run_test(cls, *,dataset, client_count=3, **kwargs):
        request_id = f"{cls.__name__}"
        tester = cls(client_count, request_id, dataset=dataset, **kwargs)
        tester._run_test()

    def _run_test(self):
        # 1. Start gRPC server directly (so we can stop it after the test)
        from exaflow.protos.aggregation_server.aggregation_server_pb2_grpc import (
            add_AggregationServerServicer_to_server,
        )
        grpc_server = grpc.server(
            grpc_futures.ThreadPoolExecutor(max_workers=config.max_grpc_connections)
        )
        add_AggregationServerServicer_to_server(AggregationServer(), grpc_server)

        health_servicer = health.HealthServicer()
        health_pb2_grpc.add_HealthServicer_to_server(health_servicer, grpc_server)
        health_servicer.set("Aggregation", health_pb2.HealthCheckResponse.SERVING)

        grpc_server.add_insecure_port(f"0.0.0.0:{config.port}")
        grpc_server.start()

        try:
            self._configure_server()
            time.sleep(2)
            # 2. Run centralized computation
            global_output = self.centralized_computation(self.dataset.get_global_dataset(),**self.kwargs)
            # 3. Run federated computation
            federated_outputs = {}
            with ThreadPoolExecutor(max_workers=self.num_of_workers) as executor:
                # Submit all tasks
                futures = []
                for client_id in range(self.num_of_workers):
                    local_data = self.dataset.get_local_dataset(client_id, self.num_of_workers)
                    future = executor.submit(
                        self._run_client,
                        f"Client_{client_id}",
                        local_data,
                        **self.kwargs
                    )
                    future.client = client_id
                    futures.append(future)

                # Collect results as they complete
                for future in as_completed(futures):
                    try:
                        result = future.result()  # This gets the return value
                        federated_outputs[future.client]=result
                    except Exception as e:
                        print(f"Task failed: {e}")
            self.compare(federated_outputs,global_output,**self.kwargs)
        finally:
            # 4. Shutdown server
            grpc_server.stop(grace=5)

    @staticmethod
    @abstractmethod
    def federated_computation(client, local_dataset,**kwargs):
        """Implement the federated logic here."""
        pass

    @staticmethod
    @abstractmethod
    def centralized_computation( centralized_dataset,**kwargs):
        """Implement the centralized ground-truth logic here."""
        pass

    @staticmethod
    @abstractmethod
    def compare(federated_output, global_output,**kwargs):
        """Compare federated vs centralized results."""
        pass