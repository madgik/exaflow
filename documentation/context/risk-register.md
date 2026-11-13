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
| Kubernetes/Helm | Deployment template changes can break service discovery, config, or volumes. | `kubernetes/templates`, `values.yaml`, prod env workflow. | Run `helm template`; use prod env tests when feasible. | Yes for runtime-affecting changes. |
| Local deployment tasks | Cleanup/removal tasks can delete containers or generated data. | `tasks.py` includes cleanup, rm-containers, data path generation. | Ask before destructive commands; inspect target paths. | Yes for destructive behavior changes. |
| SMPC/DP config | Privacy-preserving computation paths are sensitive and partly optional. | `.deployment.sample.toml`, `tasks.py`, SMPC markers. | Verify environment and tests before changes. | Yes. |
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
