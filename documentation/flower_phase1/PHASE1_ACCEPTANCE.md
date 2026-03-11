# Phase 1 Acceptance Criteria

## A. Smoke
1. Job starts from Exaflow endpoint.
2. Simulation completes at least 2 rounds without crash.
3. Terminal status is produced.

## B. Metrics
1. At least one valid `round_event` per completed round.
2. `requested_metrics` is always present.
3. `reported_metrics` follows policy:
   - RUNNING rounds: equals keys of `metrics`.
   - failures: empty allowed.
4. One `final_summary` is always emitted.
5. Timestamps are serialized in UTC with `Z` suffix.

## C. Artifacts
1. `artifact_dir` resolves and is writable.
2. Checkpoints are saved when configured.
3. On `COMPLETED`, `artifacts.final_checkpoint_ref` exists.

## D. Failure/Timeout/Cancel
1. On `FAILED/CANCELLED/TIMEOUT`, `error` exists in final summary.
2. `artifacts` is optional for non-completed final states.
3. Partial/empty metrics payloads are accepted for failed terminal states.
4. `CANCELLED` is best-effort stop and still emits terminal summary.

## E. Contract Strictness
1. Unknown RunConfig fields are rejected.
2. Invalid combinations (e.g., `local_steps` + `epochs`) are rejected.
3. Invalid `runtime != simulation` is rejected in Phase 1.
4. Duplicate `request_id` follows idempotency policy (no duplicate run creation).

## F. Non-Goals
1. No real hospital connectors.
2. No SuperLink/SuperNodes deployment runtime.
3. No production security/compliance hardening in Phase 1.
