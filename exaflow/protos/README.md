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

## mTLS authentication and `worker_ids`

`ConfigureRequest.worker_ids` is the allow-list of worker identities that may
contribute to a context. The controller sends one id per selected worker. When
the aggregation server runs with mTLS client authentication enabled:

- `Configure` and `Cleanup` accept only a certificate-bound controller identity.
- `Aggregate` and `Unregister` accept only a certificate-bound worker identity,
  and the worker id from its certificate must appear in `worker_ids`.
- `worker_ids` is required on `Configure`, must not contain blanks or
  duplicates, and must match `num_of_workers` when both are non-empty.

Identity comes from exactly one URI SAN (`peer_uri`) in the client certificate:
`exaflow://<federation>/controller` or
`exaflow://<federation>/worker/<worker_id>`. A peer without a valid identity is
`UNAUTHENTICATED`; a valid identity with the wrong role, the wrong federation,
or a worker id outside the allow-list is `PERMISSION_DENIED`. A second or
replayed submission for the same worker and step is `ALREADY_EXISTS` without
failing the whole context. Calling `Unregister` after submitting to an
in-progress step returns `FAILED_PRECONDITION`. The `x-aggregation-owner` token
is still required on `Cleanup`. When TLS is enabled, `require_client_auth` and
`FEDERATION` are both required.

gRPC health probes are served on a separate plaintext `HEALTH_PORT` when it is
configured, so kubelet can probe a TLS-only server without reaching any
`AggregationServer` RPC. The service exposes only the main aggregation port.
