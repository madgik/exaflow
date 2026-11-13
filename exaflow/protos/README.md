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

## Execution context compatibility

Aggregation RPCs carry `request_id` for logical request tracing and a required
`context_id` for the execution attempt. Configure, Aggregate, Unregister, Cleanup,
and cancellation tombstones use `context_id` as their state key. UDF system
arguments carry the same context to every worker; logs retain the request ID.
Controller, workers, and aggregation server must be upgraded together. Older
clients that omit the context are rejected; there is no request-ID fallback.
Ownership-token checks on Cleanup remain mandatory.
