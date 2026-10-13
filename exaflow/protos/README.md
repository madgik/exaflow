## gRPC stub generation

From the repository root, regenerate the Python stubs with:

Aggregation server:

```bash
uv run python -m grpc_tools.protoc \
  -I . \
  --python_out=. \
  --grpc_python_out=. \
  exaflow/protos/aggregation_server/aggregation_server.proto
```

Worker service:

```bash
uv run python -m grpc_tools.protoc \
  -I . \
  --python_out=. \
  --grpc_python_out=. \
  exaflow/protos/worker/worker.proto
```

## RunUdf payloads

`RunUdfRequest.kw_args`, `RunUdfRequest.system_args`, and `RunUdfResponse.result`
are UTF-8 JSON bytes produced by `exaflow/udf_serialization.py`. JSON keeps
integers exact and preserves dict order, which `google.protobuf.Struct` did not.
Fields 3 and 4 of `RunUdfRequest` (the former `Struct` args) are reserved.
