# Flower future work and open items

Updated: 2026-11-18. This is the canonical follow-up register. Nothing in the
post-initial backlog is a prerequisite for the initial integration. The
[initial plan](plan.md), [design](design.md), and [decision log](decisions.md)
retain the chosen baseline; the [failure board](failure-board.md) describes its
error/restart behavior.

## After the initial implementation

| ID | Work | Status / trigger | Questions to resolve when starting it |
| --- | --- | --- | --- |
| FW01 | Role-aware fit/transform preprocessing | Explicitly deferred by the user, C49/C50; separate [PRD](prd/role-aware-preprocessing.md) | Hook/specification contract, fitted-state representation, first supported step variants, and mapping local fitted state to validation-only workers |
| FW02 | Enhanced Flower participant selection and complete-round enforcement | Deferred under C44 / R19 | Require all selected nodes, permit a quorum, or use an algorithm-specific partial-participation policy? Define which failures invalidate a round and test unrelated/disappearing nodes |
| FW03 | Independent native/Flower eligibility | Deferred under C41 / R25; revisit if operational failures demonstrate the need | How to advertise each engine's availability without excluding healthy native execution or complicating discovery unnecessarily? |
| FW04 | Periodic cleanup of expired prepared files on idle pods | Candidate enhancement: the periodic-sweep portion of R22 remains open | Sweep cadence and ownership, staging cleanup, and safe expiry relative to live readers. Initial cleanup/permissions remain required below |
| FW05 | Round/progress reporting | Excluded from the initial small coordinator under C25 | Required progress fields, source of truth, polling behavior, and whether partial rounds have useful user-facing meaning |
| FW06 | Pooled/global AUC across validation workers | Excluded by initial C45, which returns per-worker AUCs and pooled log loss | Choose and validate an evaluation method, accuracy, communication, and privacy requirements. Averaging local AUCs is not pooled AUC |

FW01 offers framework capabilities; the algorithm developer chooses the
statistical workflow. Independent evaluation fits on training datasets and
transforms **both training and validation** with that fitted state. The initial
integration does not add these phases or silently change existing preprocessing.

FW02 does not defer basic data/context/role isolation, required privacy checks, or
reporting the actual evaluation participants. FW03 keeps the current whole-pod
exclusion until there is a reason to change it. FW04 does not defer file integrity,
expiry checks before use, shared-container permissions, or bounded storage.

## Longer-term work, not committed features

These were explicitly excluded from the initial scope. Reconsider each as a
separate product decision rather than assuming it will be implemented.

| Area | Existing decision | Needed before starting |
| --- | --- | --- |
| Persistent executions / restart recovery | C14 excludes it; C40 resets old work and permits RabbitMQ fresh-attempt redelivery | End-to-end controller persistence/ownership and reply delivery, coordinator/runtime state, and model checkpoints where training is to resume. Saving only SuperLink state is insufficient |
| Stored models and inference | C02 returns metrics only | Model ownership, storage/access/lifetime, inference input/output contracts, and disclosure review |
| Random train/validation row splitting and cross-validation | C06 uses separate whole datasets | Explicit splitting unit, reproducibility, privacy after each split, and fit/transform behavior within each fold; do not assume dataset-code separation proves distinct patients |

## Initial implementation: open details that must still be completed

No further product decision is required before starting the agreed implementation.
These are implementer-owned details and release checks, **not** optional future
features or a questionnaire for the user. Resolve them within the chosen scope,
record the concrete behavior, and test it as the work proceeds. Raise material
scope/behavior changes with the user. Privacy approval requires the designated
human reviewer before release, not before development starts.

| Item | Owner | Initial requirement / current recommendation |
| --- | --- | --- |
| XGBoost defaults, fixed safeguards, and privacy approval (R06/C23) | Algorithm developer and privacy reviewer | Finalize the allowed settings and the model-disclosure review before release; minimum-row checks remain mandatory |
| Exact dependency lock and runtime API/probe verification (R07/R12) | Flower runtime developer | Resolve/test the pinned release stack, direct service commands, health checks, subprocess behavior, resource settings, and executor capacity for native execution plus Flower preparation |
| Coordinator restart cleanup and execution-loss mechanism (R14) | Controller/runtime developer | Implement and verify an instance-aware loss signal and bounded orphan termination before accepting replacement work. This is required to satisfy C40/C48, not execution recovery |
| Confirmed, mandatory AMQP forwarding (R16) | P2 transport developer | Reuse the existing publication/settlement guarantees; test unroutable/negative/uncertain forwarding outcomes and terminal error delivery |
| Manifest, categorical encoding, and parameter-floor handoff (R23) | Controller/worker/runtime developers | Specify ordered features/types, role/context identity, outcome mapping, counts, expiry, and matching readers/writers. Keep raw data local and Flower privacy checks out of scope |
| File cleanup baseline and permissions (R09, part of R22) | Worker/runtime developers | Reader deletion at normal completion, cleanup on write failure, expiry checks before use, expired leftovers removed at next preparation, shared UID/GID/mode contract, and storage bounds. Idle files are not guaranteed wall-clock deletion until FW04 exists |
| Initial participant/result contract | Algorithm/runtime developer | Route by prepared role, retain actual evaluator IDs/counts, and document the initial Flower strategy's partial-failure behavior until FW02 |
| One-class local validation AUC | Algorithm developer | AUC is undefined for one observed class. Current recommendation: `auc: null` with `single_class` reason, while retaining valid log loss/counts. Choose/document this edge-case behavior before finalizing the result schema; do not create a new per-class privacy threshold |

The node-authentication simplification suggestion R17 does not block work: C29's
existing TLS/node-authentication choice remains the baseline unless explicitly
changed. C47's `Recreate`, C48's capacity rejection, C42's status-read-only retry
exception, and C45/C46's evaluation/outcome choices are already settled.

## Closed choices, not a future-work queue

Do not revive cross-version compatibility, specification hashes, coordinator gRPC,
legacy request/result aliases, Flower-side duplicate privacy checks, or an extra
reverse proxy as prerequisites. They were deliberately excluded. P1–P3 remain
separate **initial prerequisites**, not work deferred by this register.

The deferred fit/transform extension has one known initial limitation: current
preprocessing may fit on validation rows together with training rows. Initial
evaluation documentation must describe that behavior and must not promise a
training-only fitted holdout pipeline.
