# Risk Register

| Area | Risk | Evidence | Agent guidance | Human review required? |
| --- | --- | --- | --- | --- |
| Analysis request/spec API | Client form shape and validation depend on runtime specifications. | `GET /specifications/*`, `POST /analysis`, `AnalysisRequestDTO`, `documentation/api-specification.md`. | Treat spec changes as public API changes; update docs and tests. | Yes for breaking changes. |
| Dynamic algorithm discovery | Import path changes can remove algorithms or double-register UDFs. | `exaflow/__init__.py` imports from `EXAREME3_ALGORITHM_FOLDERS`. | Preserve idempotent import behavior; run discovery/validator tests. | Yes for discovery refactors. |
| UDF registry | Missing or duplicate UDF registrations break worker execution. | `@exareme3_udf` registry use in algorithm wrappers and worker UDF service. | Add focused tests and run algorithm validation. | Yes for registry changes. |
| Worker privacy checks | Minimum row count protects local data. | `worker_config.privacy.minimum_row_count` used in UDF service. | Do not lower or bypass checks without explicit review. | Yes. |
| DuckDB data loading | Data path, schema, and CSV parsing affect all worker execution. | `duck_db_csv_loader`, worker config, `tests/test_data`. | Run focused loader tests; avoid hardcoded local paths. | Yes for broad loader changes. |
| Aggregation lifecycle | Failure to configure or cleanup can leak request state or corrupt results. | `ControllerAggregationClient`, aggregation server `Configure/Aggregate/Cleanup`. | Preserve cleanup in `finally`; test aggregation clients. | Yes for protocol/lifecycle changes. |
| Protobuf contracts | Generated code and services must stay compatible. | `exaflow/protos/*/*.proto` and generated files. | Regenerate files and run gRPC tests. | Yes. |
| Kubernetes/Helm | Deployment template changes can break service discovery, config, or volumes. | `kubernetes/templates`, `values.yaml`, prod env workflow. | Run `helm template`; use prod env tests when feasible. | Yes for runtime-affecting changes. |
| Local deployment tasks | Cleanup/removal tasks can delete containers or generated data. | `tasks.py` includes cleanup, rm-containers, data path generation. | Ask before destructive commands; inspect target paths. | Yes for destructive behavior changes. |
| SMPC/DP config | Privacy-preserving computation paths are sensitive and partly optional. | `.deployment.sample.toml`, `tasks.py`, SMPC markers. | Verify environment and tests before changes. | Yes. |
| CI secrets and publishing | Workflows push images/packages and use credentials. | Publish workflows and Docker registry login steps. | Do not expose secrets; keep publish changes reviewed. | Yes. |
| `exadata-validator` package | Separate package release can break external CLI users. | Nested package workflow builds and smoke-tests wheel. | Run nested tests/build for package changes. | Yes for release/CLI changes. |
| Test fixtures | Expected JSON fixtures define behavior for algorithms. | `tests/prod_env_tests/expected`, validation expected fixtures. | Update fixtures only with justified behavior changes. | Sometimes. |
| Dependency updates | Scientific stack changes can alter numerical results. | NumPy/SciPy/pandas/scikit/statsmodels dependencies. | Run focused numerical tests and compare fixtures. | Yes for major updates. |
