# flowertune_llm_medical (Phase 1)

This package provides simulation-first Flower runtime wiring for Exaflow.

Current implementation scope:
- strict RunConfig validation + env mapping contract
- lightweight federated training runtime (tiny adapter model for smoke runs)
- backend switch in contract: `model.backend = tiny | hf_peft` (`tiny` default)
- dataset adapter from worker CSVs when `inputdata.x`/`inputdata.y` are provided
- synthetic fallback partition only when no CSV dataset input is provided
- FedAvg strategy with checkpoint cadence and final summary reporting
- final summary payload validation against MetricsOut policy before posting
- checkpoint semantics: `final_checkpoint_ref` is always a filesystem path in completed runs

Backend behavior:
- `tiny`: full Phase 1 smoke runtime.
- `hf_peft`: HF model/tokenizer load + LoRA attach (PEFT), adapter-only local training/evaluation, adapter tensor exchange.
  - Requires: `torch`, `transformers`, `peft` (preflight-validated).

Play mode (local simulation with prints):
- Run from project root:
  `python -m exaflow.algorithms.flower.flowertune_llm_medical.main --backend tiny --rounds 2 --clients 2 --local-steps 1`
- You will see per-client and per-round metrics in stdout, then final metrics and `final_checkpoint_ref`.

Execution scope policy:
- Phase 1 supports a single active Flower job at a time (`algorithm_execution_lock`).
- Flower controller endpoints are request-scoped via `request_id` for this algorithm.

Planned next scope:
- replace tiny adapter runtime with full LLM/PEFT (LoRA) runtime
- round-level MetricsOut event emission
- deployment runtime support (SuperLink/SuperNodes)
