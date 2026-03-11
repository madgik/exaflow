# Validation Spec - Phase 1

## Validation Layers
1. Schema Validation (Pydantic).
2. Service Validation (Exaflow business/runtime checks).
3. Runtime Validation (Flower app startup/runtime checks).

## A. Pydantic Validation (Request `parameters`)
1. `extra = "forbid"` on all models.
2. `runtime` must be `simulation`.
3. `schema_version` must be `1.1`.
4. `federation.num_rounds >= 1`, `num_clients >= 1`.
5. `min_fit_clients <= num_clients`.
6. `min_evaluate_clients <= num_clients`.
7. `min_available_clients <= num_clients`.
8. If `evaluation.enabled = false`: `fraction_evaluate = 0`, `min_evaluate_clients = 0`.
9. `local_steps XOR epochs` (exactly one set).
10. `fp16` and `bf16` cannot both be `true`.
11. `seed >= 0`.
12. `job_timeout_sec >= round_timeout_sec`.
13. For `partitioner = iid`: `dataset.num_partitions = federation.num_clients`.
14. If `peft.enabled = true`: `method`, `r`, `alpha` are required.
15. Metrics allowlist for Phase 1: `loss`, `perplexity`.

## B. Service Validation (Exaflow)
1. Resolve `artifact_dir` template (`${request_id}`).
2. Verify writable artifact path.
3. Enforce capacity policy:
   - max allowed `num_clients` for simulation,
   - max allowed `model.max_seq_length`,
   - allowed quantization modes,
   - timeout upper bounds,
   - optional GPU requirement policy.
4. Normalize env map with canonical serialization rules.
5. Enforce idempotency policy for duplicate `request_id` (return existing `job_id`).
6. Enforce event invariants at ingestion:
   - `round <= rounds_total`,
   - `rounds_completed <= rounds_total`,
   - `clients.participated <= clients.expected`,
   - `clients.failed <= clients.expected`,
   - `clients.participated + clients.failed <= clients.expected`,
   - if present: `best_round <= rounds_completed`.
7. Enforce metrics consistency policy:
   - RUNNING round event: `reported_metrics == keys(metrics)`.
   - FAILED/CANCELLED/TIMEOUT: empty metrics/report sets allowed.

## C. Runtime Validation (Flower app)
1. Model/tokenizer load success.
2. Dataset load and partitioning success.
3. Quantization compatibility.
4. LoRA target modules validity.
5. Early failure produces structured `ErrorInfo`.
