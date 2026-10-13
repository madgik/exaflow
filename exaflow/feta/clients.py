from threading import Lock

import grpc

from exaflow.feta.lifecycle import ContextClosedError
from exaflow.feta.lifecycle import ResourceLimitError
from exaflow.feta.operations import decode_tensor
from exaflow.protos.feta import feta_pb2 as pb
from exaflow.protos.feta import feta_pb2_grpc as rpc
from exaflow.worker_communication import BadUserInput
from exaflow.worker_communication import InsufficientDataError


class FetaRpcClient:
    def __init__(self, address, stub_class, timeout=120):
        if not isinstance(address, str) or not address:
            raise ValueError("FETA requires an explicit gRPC service address.")
        self.timeout = timeout
        self.channel = grpc.insecure_channel(
            address,
            options=[
                ("grpc.max_send_message_length", 100 * 1024 * 1024),
                ("grpc.max_receive_message_length", 100 * 1024 * 1024),
            ],
        )
        self.stub = stub_class(self.channel)
        self.lock = Lock()
        self.pending = set()
        self.stopping = False

    def call(self, name, request):
        with self.lock:
            if self.stopping and name != "Cleanup":
                raise ContextClosedError("FETA RPC client is closing.")
            future = getattr(self.stub, name).future(request, timeout=self.timeout)
            self.pending.add(future)
        try:
            return future.result()
        except grpc.FutureCancelledError as exc:
            raise ContextClosedError("FETA RPC was cancelled.") from exc
        except grpc.RpcError as exc:
            message = exc.details() or str(exc.code())
            code = exc.code()
            if code == grpc.StatusCode.INVALID_ARGUMENT:
                raise BadUserInput(message) from exc
            if code == grpc.StatusCode.FAILED_PRECONDITION:
                raise InsufficientDataError(message) from exc
            if code == grpc.StatusCode.NOT_FOUND:
                raise ContextClosedError(message) from exc
            if code == grpc.StatusCode.RESOURCE_EXHAUSTED:
                raise ResourceLimitError(message) from exc
            if code == grpc.StatusCode.DEADLINE_EXCEEDED:
                raise TimeoutError(message) from exc
            raise RuntimeError(f"FETA RPC {name} failed: {message}") from exc
        finally:
            with self.lock:
                self.pending.discard(future)

    def cancel(self):
        with self.lock:
            self.stopping = True
            pending = list(self.pending)
        for future in pending:
            future.cancel()

    def close(self):
        self.cancel()
        self.channel.close()


class FetaWorkerClient(FetaRpcClient):
    def __init__(self, worker_id, address, timeout=120):
        super().__init__(address, rpc.FetaWorkerStub, timeout)
        self.worker_id = worker_id

    def initialize(self, request):
        return self.call("InitializeContext", request)

    def execute(self, request):
        return self.call("Execute", request)

    def cleanup(self, context_id):
        self.call("Cleanup", pb.ContextRequest(context_id=context_id))


class FetaAggregationClient(FetaRpcClient):
    def __init__(self, address, timeout=120):
        super().__init__(address, rpc.FetaAggregationStub, timeout)

    def prepare(self, request):
        self.call("Prepare", request)

    def submit(self, request):
        self.call("Submit", request)

    def fetch(self, context_id, step):
        result = self.call("Fetch", pb.StepRequest(context_id=context_id, step=step))
        return decode_tensor(result, 100 * 1024 * 1024)

    def cleanup(self, context_id):
        self.call("Cleanup", pb.ContextRequest(context_id=context_id))
