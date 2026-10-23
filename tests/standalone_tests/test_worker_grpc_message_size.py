from exaflow.controller.worker_client.app import WorkerClient
from exaflow.protos.worker import worker_pb2
from exaflow.protos.worker import worker_pb2_grpc
from exaflow.worker import grpc_server

FIVE_MIB = 5 * 1024 * 1024


class _LargeResultService(worker_pb2_grpc.WorkerServiceServicer):
    def RunUdf(self, request, context):
        assert len(request.udf_registry_key) == FIVE_MIB
        return worker_pb2.RunUdfResponse(result=b"x" * FIVE_MIB)


def test_run_udf_messages_above_default_grpc_limit_round_trip():
    server = grpc_server._create_grpc_server(max_workers=1)
    worker_pb2_grpc.add_WorkerServiceServicer_to_server(_LargeResultService(), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    client = WorkerClient(f"127.0.0.1:{port}")
    try:
        response = client.call(
            "RunUdf",
            worker_pb2.RunUdfRequest(udf_registry_key="k" * FIVE_MIB),
            timeout=30,
        )
    finally:
        client.close()
        server.stop(0)

    assert len(response.result) == FIVE_MIB
