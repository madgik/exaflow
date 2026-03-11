# Design Decisions - Phase 1 (Simulation-First)

## Scope
The Phase 1 implementation provides federated training orchestration through Exaflow + Flower using simulation runtime.
It does not include production hospital deployment, SuperLink/SuperNodes, or compliance hardening.

## Core Architecture Split
1. Exaflow = control plane.
2. Flower = training/aggregation engine.
3. Flower App = algorithm-specific training logic (flowertune_llm_medical).

## Source of Truth Policy
1. For Phase 1, source of truth for training data/runtime knobs is `parameters.dataset`.
2. `inputdata` remains for compatibility and metadata.

## Integration Strategy
1. Do not vendor the full `adap/flower` repository.
2. Integrate only the Flower app code as an Exaflow algorithm package.
3. Keep Flower framework as a dependency.

## Runtime Policy
1. Only `runtime = simulation` is supported in Phase 1.
2. The same API shape will extend to Phase 2 deployment runtime.
3. `dataset` block semantics may change under `runtime=deployment` (local connectors instead of simulation partitioning).

## Exaflow Boundary
1. Exaflow does not manage model internals (LoRA weights, state dicts).
2. Exaflow handles config, lifecycle status, metrics, and artifact references.

## Contract Versioning
1. `schema_version = "1.1"` for RunConfig and MetricsOut in this freeze.
2. Backward-incompatible changes require a new major schema version.
3. Backward-compatible additions may remain in the same major series.

## Serialization Policy
1. Boolean env values: `"true"` / `"false"`.
2. List env values: JSON string.
3. Optional `None`: env var omitted.
4. `${request_id}` interpolation is resolved by Exaflow before dispatch.
