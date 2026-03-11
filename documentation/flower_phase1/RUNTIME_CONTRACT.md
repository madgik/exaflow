# Runtime Contract - Phase 1

## Trigger
`POST /algorithms/flowertune_llm_medical`

## Request Shape
1. Exaflow standard `AlgorithmRequestDTO`.
2. RunConfig is provided inside `parameters`.

## RunConfig -> Env Mapping Rules
1. Flatten nested config into env vars.
2. booleans -> `"true"` / `"false"`.
3. lists -> JSON strings.
4. `None` -> omit env var.
5. Canonical full naming for env vars (e.g., `MIN_EVALUATE_CLIENTS`).

## Lifecycle
1. `SUBMITTED`
2. `RUNNING`
3. Terminal: `COMPLETED` | `FAILED` | `CANCELLED` | `TIMEOUT`

## Concurrency Scope (Phase 1)
1. Single active Flower job is supported at a time.
2. Flower helper endpoints are request-scoped via `request_id` query parameter.

## Event Flow
1. Exaflow receives and validates request.
2. Exaflow creates job id and resolves artifact path.
3. Exaflow dispatches Flower simulation run.
4. Flower emits `round_event` payloads.
5. Flower emits one terminal `final_summary` payload.
6. Exaflow persists status, metrics, and artifacts references.

## MetricsOut Status Semantics
1. `round_event.status` is only `RUNNING` or `FAILED`.
2. `COMPLETED` appears only in `final_summary.status`.
3. `round_event.artifacts` object is always present; `checkpoint_ref` is optional.

## Observability Fields
1. `job_id` is required and the primary correlation key.
2. `request_id` is optional but recommended.
3. `correlation_id` is optional.
4. `algorithm_name`, `runtime`, `timestamp` are required.
5. Timestamps must be UTC with `Z` suffix.

## Error Contract
1. Error payload shape: `code`, `message`, optional `details`.
2. `details` is intentionally open for runtime-specific diagnostics.
3. Canonical Phase 1 error codes:
   - `VALIDATION_ERROR`
   - `ARTIFACT_PATH_ERROR`
   - `MODEL_LOAD_ERROR`
   - `DATASET_LOAD_ERROR`
   - `QUANTIZATION_ERROR`
   - `PEFT_CONFIG_ERROR`
   - `ROUND_TIMEOUT`
   - `JOB_TIMEOUT`
   - `RUNTIME_ERROR`
   - `CANCELLED_BY_USER`

## Defaults Note
`default` values in JSON Schema are informational only. Defaults are applied by producer/service logic, not by JSON Schema validators.

## Cancellation Semantics
Cancellation is best-effort stop for Phase 1 simulation. A terminal `final_summary` with `status=CANCELLED` is required.
