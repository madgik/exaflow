# Risk Register

| Area | Risk | Evidence | Agent guidance | Human review required? |
| --- | --- | --- | --- | --- |
| Analysis request/spec API | Client form shape and validation depend on runtime specifications. | `GET /specifications/*`, `POST /analysis`, `AnalysisRequestDTO`, `documentation/api-specification.md`. | Treat spec changes as public API changes; update docs and tests. | Yes for breaking changes. |
| Dynamic algorithm discovery | Import path changes can remove algorithms or double-register UDFs. | `exaflow/__init__.py` imports from `EXAREME3_ALGORITHM_FOLDERS`. | Preserve idempotent import behavior; run discovery/validator tests. | Yes for discovery refactors. |
| UDF registry | Missing or duplicate UDF registrations break worker execution. | `@exareme3_udf` registry use in algorithm wrappers and worker UDF service. | Add focused tests and run algorithm validation. | Yes for registry changes. |
| Worker privacy checks | Minimum row count protects local data. | `worker_config.privacy.minimum_row_count` used in UDF service. | Do not lower or bypass checks without explicit review. | Yes. |
| DuckDB data loading | Data path, schema, and CSV parsing affect all worker execution. | `duck_db_csv_loader`, worker config, `tests/test_data`. | Run focused loader tests; avoid hardcoded local paths. | Yes for broad loader changes. |
| Aggregation lifecycle | Failure to configure or cleanup can leak request state, corrupt results, or leave workers blocked until the aggregation timeout. | `ControllerAggregationClient`, aggregation server `Configure/Aggregate/Cleanup`. | Preserve cleanup in `finally`: it aborts workers still waiting when another worker fails. Contexts that never get `Cleanup` are evicted after `STALE_CONTEXT_TTL_SECONDS`. Test status codes against a real gRPC server (`test_aggregation_server_grpc.py`), not only the inline stub. | Yes for protocol/lifecycle changes. |
| Protobuf contracts | Generated code and services must stay compatible. | `exaflow/protos/*/*.proto` and generated files. | Regenerate files and run gRPC tests. | Yes. |
| Aggregation mTLS identity | A mis-issued or stolen certificate could claim another role, federation, or worker id. | `exaflow/grpc_tls.py` (URI SAN schema and `extract_peer_identity`), `ConfigureRequest.worker_ids`, `AggregationServer._authorize`. | Keep `require_client_auth` enabled in production, issue exactly one `exaflow://` URI SAN per certificate, and keep the `worker_ids` allow-list authoritative. Test rejections against a real mTLS server (`test_aggregation_mtls.py`). | Yes, for certificates, role policy, and rollout. |
| Kubernetes/Helm | Deployment template changes can break service discovery, config, or volumes. | `kubernetes/templates`, `values.yaml`, prod env workflow. | Run `helm template`; use prod env tests when feasible. | Yes for runtime-affecting changes. |
| Local deployment tasks | Cleanup can delete generated data. | `tasks.py` includes cleanup and data path generation. | Ask before destructive commands; inspect target paths. | Yes for destructive behavior changes. |
| CI secrets and publishing | Workflows push images/packages and use credentials. | Publish workflows and Docker registry login steps. | Do not expose secrets; keep publish changes reviewed. | Yes. |
| `exadata-validator` package | Separate package release can break external CLI users. | Nested package workflow builds and smoke-tests wheel. | Run nested tests/build for package changes. | Yes for release/CLI changes. |
| Test fixtures | Expected JSON fixtures define behavior for algorithms. | `tests/prod_env_tests/expected`, validation expected fixtures. | Update fixtures only with justified behavior changes. | Sometimes. |
| Dependency updates | Scientific stack changes can alter numerical results. | NumPy/SciPy/pandas/scikit/statsmodels dependencies. | Run focused numerical tests and compare fixtures. | Yes for major updates. |

## SQL and CI hardening

Worker loader SQL identifiers are quoted, including metadata-derived model names.
Validator bounds must be finite JSON numbers (booleans and numeric strings are
rejected); absent or null bounds are unset. Bounds are passed as SQL parameters.

External GitHub Actions use verified full commit pins with version comments.
Update the pin and comment together after verifying the upstream release.
Workflows use read-only repository permissions and disable checkout credential
persistence. Publishing workflows build without restoring or saving CI caches;
test caches remain enabled. Controller, worker, and staging aggregation images
retain separate publishing jobs so one failure does not skip other images. Publishing credentials and triggers are unchanged.
Rollback is a revert of these changes, restoring the corresponding exposures.

## RabbitMQ analysis transport

Broker persistence adds storage for analysis results and dead letters. Bound
message size, queue growth, and retention; provide credentials through existing
configuration/Secrets and keep URLs out of logs. AMQP is disabled by default and
normal deployments provide an external broker. HTTP and AMQP share one analysis
capacity limit; other HTTP routes remain independent.

A sustained queue backlog can reject HTTP analyses with 503. Reply failures
must remain isolated from the consuming channel. Publish on a separate AMQP
connection so a memory or disk alarm cannot block acknowledgements. Finished
replies retained for redelivery are bounded to five minutes, 128 entries, and
64 MiB; eviction or a controller restart still permits recomputation.
Acknowledge only after confirmed publication, and keep reply bodies and
connection URLs out of diagnostics.

Delivery is at least once, with recomputation and duplicate replies allowed.
Preserve logical request IDs and isolate aggregation state using attempt context
IDs, including late calls, cancellation tombstones, and cleanup ownership checks.
The internal protocol change requires coordinated controller/worker/aggregation
rollout. Consumer tasks require explicit draining before existing shutdown cancels
RPCs; cancelling asyncio tasks alone does not stop executor threads.

Real-broker recovery tests may restart only their dedicated kind test broker and
replace the test controller. They require the explicit test recovery environment
flag and must never target shared deployments. See
`documentation/rabbitmq-analysis.md` for configuration, tests, and rollback.

## Aggregation mTLS and certificate-bound identity

The aggregation server, controller, and workers can authenticate each other with
mutual TLS. Certificates carry exactly one URI SAN identifying the caller:
`exaflow://<federation>/controller` or
`exaflow://<federation>/worker/<worker_id>`. The server authorizes each RPC by
role, checks the federation, and enforces the per-context `worker_ids`
allow-list submitted on `Configure`. Rejected identities are local to the call:
an outsider's `Aggregate`/`Unregister` or a repeated submission never fails the
honest workers' context. `Cleanup` still requires the ownership token.

Kubelet gRPC probes reach a separate plaintext `HEALTH_PORT` that registers only
`HealthServicer`; `AggregationServer` is never exposed there and the Service
publishes only the main port (the health port binds `0.0.0.0` for kubelet pod-IP
probes; restrict access with a NetworkPolicy if cluster-internal pods are
untrusted). Helm rejects exposing the aggregation Service outside `ClusterIP`
while TLS is disabled, and requires `health_port` to be set and distinct from
`50051` when TLS is enabled. TLS channel clients keep the DNS name for SNI/SAN
verification instead of resolving it to an IP. In Kubernetes, the controller and
worker pods mount Secrets with `fsGroup: 10001` and mode `0440` (`appuser`), and
`worker_secret` provides per-worker `<WORKER_IDENTIFIER>.crt`/`.key` files plus
`ca.crt` so every worker replica authenticates with its own `worker_id` URI SAN.

Rollout is coordinated: issue the CA and certificate Secrets first, set
`aggregation_server.tls.enabled`, then restart the aggregation server,
controller, and workers together. `worker_ids` and `num_of_workers` must agree.
Rollback disables `aggregation_server.tls.enabled` and restores plaintext
channels and the aggregate health/probe port; generated certificate files are
not modified by the application. Because this changes the Configure request and
authorization behavior, all three components must be upgraded or rolled back as
one unit.
