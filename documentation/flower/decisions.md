# Flower integration decisions

Last updated: 2026-11-18. Planning only; nothing is implemented.

**Confirmed** means the team chose it. **Recommended** means it is proposed and
not yet finalized in implementation. Technical recommendations are starting points
the implementer may resolve within the confirmed scope; they do not require another
product decision before work starts. Document and test the chosen implementation.
Privacy approval remains a human release gate. When a decision changes, update
[plan.md](plan.md) and [design.md](design.md), and move the old entry to
"Superseded" so nobody revives it by accident.

C19–C36 came from an architecture review of the first draft; the full review is in
git history (commit `73b167184`).

## Confirmed

| ID | Decision | Details |
| --- | --- | --- |
| C01 | First algorithm: binary classification with XGBoost | [XGBoost and privacy](design.md#xgboost-and-privacy) |
| C02 | Metrics only. No stored model, no inference | [Result](design.md#result) |
| C03 | Flower sidecar in each worker pod. Prepared data is handed over as Arrow files on a pod-private volume | [Inside a worker pod](design.md#inside-a-worker-pod) |
| C04 | Keep the message format platform-backend publishes. No public job API. How analyses arrive is set by C30 | [Overview](plan.md#overview) |
| C05 | Initial Flower preparation stays in Exaflow and reuses current native preprocessing unchanged | C50 defers the fit/transform extension. Document the initial joint-preprocessing limitation; [Data preparation](design.md#data-preparation) |
| C06 | Training and validation are separate whole datasets. No random row split | [Scope](plan.md#scope) |
| C07 | Validation datasets may live on multiple workers. A worker may train, validate, or do both | User clarified there is no one-validation-worker limit; [Scope](plan.md#scope) |
| C08 | Exaflow and Flower images are released together as one version | [Scope](plan.md#scope) |
| C09 | No cross-version compatibility features: no specification hashes, version negotiation, aliases, or fallbacks | [Scope](plan.md#scope) |
| C10 | Planning documents live in `documentation/flower/` | |
| C12 | HTTP/JSON between controller and coordinator. Native worker data-execution and aggregation transports remain gRPC | P1 separately removes worker-role metadata/protobuf fields; Flower adds no new worker RPC; [Coordinator API](design.md#coordinator-api) |
| C13 | Each Flower algorithm's JSON specification lives beside its code. A build step copies it into the controller, which loads it at startup | [Project layout](design.md#project-layout-and-specifications) |
| C14 | No saved execution state and no restart recovery. In-memory records with deadlines and cleanup | [Failures and cleanup](design.md#failures-and-cleanup) |
| C15 | Project directory `exaflow-flower/` | [Project layout](design.md#project-layout-and-specifications) |
| C16 | Privacy checks run only in Exaflow, before export. Flower checks only file integrity and expiry | [Data preparation](design.md#data-preparation) |
| C17 | One Flower image. Each container role uses a different start command | [Images and deployment](design.md#images-and-deployment) |
| C18 | Python package name `exaflower` | [Project layout](design.md#project-layout-and-specifications) |
| C19 | The global-worker role is removed before Flower work starts, as a separate change (prerequisite P1) | [Prerequisite](design.md#prerequisite-global-worker-removal) |
| C20 | Flower runs have their own concurrency limit (1–2), `flower.max_concurrent_runs`, using Flower queue prefetch and the coordinator admission check in C48 | [Concurrency](design.md#concurrency) |
| C21 | Flower analyses arrive through RabbitMQ, so no HTTP timeout alignment with platform-backend is needed | [Overview](plan.md#overview) |
| C22 | All Exaflow app submissions go through the coordinator, which invokes the `flwr` CLI. The CLI uses SuperLink's Control API; this application flow does not enforce coordinator-only access to that API | [Security setup](design.md#security-setup) |
| C23 | XGBoost model exchange leaks data-derived values to the coordinator and the hospitals. This is a known privacy risk that needs sign-off | [Who sees what](design.md#who-sees-what) |
| C25 | Smaller coordinator: no progress reporting, no catalog endpoint, no explicit release call. SuperLink runs in its own container | [Coordinator API](design.md#coordinator-api) |
| C26 | Use Flower's built-in features: `node_config` for worker identity, `run_config` for parameters, simulation mode for tests. Pin the exact Flower version | [How Exaflow uses Flower](design.md#how-exaflow-uses-flower) |
| C27 | A new `FlowerStrategy`, selected by `type: flower`, prepares/exports inputs with `flower_prepare` on the existing `RunUdf` path after unchanged native preprocessing. No new gRPC operation or protobuf change | No fit/transform orchestration in the initial integration; [Data preparation](design.md#data-preparation) |
| C28 | One `Exareme3Strategy` with a shared `aggregation_context` helper replaces the two Exareme3 strategies. `FlowerStrategy` uses the helper around data preparation only (prerequisite P3) | [One Exareme3 strategy](design.md#prerequisite-one-exareme3-strategy) |
| C29 | NetworkPolicies, TLS, SuperNode authentication, and sidecar hardening; no signed-app verification. Under C39, permitted pods share access to the SuperLink listener, so coordinator-only submission is an application convention rather than enforced route isolation | [Security setup](design.md#security-setup) |
| C30 | Deployment setting `analysis_intake: rabbitmq \| http`, default `rabbitmq`. In `rabbitmq` mode `POST /analysis` is off; specification and metadata endpoints stay on. `http` mode is for testing/manual runs, with no controller intake limit; C48 still guards Flower starts (prerequisite P2) | [Intake and queues](design.md#prerequisite-analysis-intake-and-queues) |
| C31 | Three queues: intake `exaflow.analysis`, `exaflow.analysis.exareme3` (prefetch `max_concurrent_experiments`), `exaflow.analysis.flower` (prefetch `flower.max_concurrent_runs`). The intake name deliberately changes the current default; request bodies, correlation IDs, and replies keep their format. All queues share durability, dead-lettering, and redelivery settings (prerequisite P2) | [Intake and queues](design.md#prerequisite-analysis-intake-and-queues) |
| C32 | The router rejects a missing or disabled algorithm and an unreadable message, using the existing reply code. All other validation happens only in the executors | [Intake and queues](design.md#prerequisite-analysis-intake-and-queues) |
| C33 | The router acks an intake message only after the broker confirms the forward. A crash in between can cause a duplicate, which the current contract allows | [Intake and queues](design.md#prerequisite-analysis-intake-and-queues) |
| C34 | `analysis_slot`, the controller concurrency semaphores, and the HTTP intake "capacity exceeded" 503 are deleted. Queue prefetch controls delivery; C48 adds a coordinator check for live Flower work | [Intake and queues](design.md#prerequisite-analysis-intake-and-queues) |
| C35 | At least one prod-environment suite runs in `rabbitmq` mode | [Intake and queues](design.md#prerequisite-analysis-intake-and-queues) |
| C36 | Prerequisites P1, P2, and P3 land before Flower work, as separate PRs that can run in parallel | User reaffirmed this ordering because it simplifies Flower development; [Prerequisites](plan.md#prerequisites) |
| C37 | Every requested worker/input role must meet the privacy minimum independently; otherwise fail the entire Flower analysis. Never silently exclude insufficient training input. A new request may explicitly omit that training dataset and retain the worker's valid validation input | User clarified the 8-training/100-validation-row example; [Data preparation](design.md#data-preparation) |
| C38 | During the prerequisite transition, known Flower analyses return an explicit terminal unavailable error until the replacement is ready. Never invoke the old global-worker-dependent strategy | User requested the intermediate error; [Intake and queues](design.md#prerequisite-analysis-intake-and-queues) |
| C39 | Keep the initial protected Kubernetes deployment without a reverse proxy, trusting permitted workloads/operators with the shared SuperLink listener | User confirmed this approach after the CLI/API clarification. Retain NetworkPolicies/TLS and the existing C29 node-authentication choice; there is no enforced installed-app allowlist against direct submission by a permitted caller |
| C40 | On controller startup, reset owned Flower work before starting the Flower executor consumer | User accepted the startup reset. Cancel partial work; after cleanup, unacknowledged AMQP requests can run as fresh attempts. No saved execution recovery |
| C41 | Whole-pod exclusion: if either native Exareme3 or Flower is unready/unhealthy, the worker pod is ineligible for analyses | User reaffirmed the simpler policy; revisit independent eligibility only if the failure coupling is observed. Use normal pod readiness and component probes; do not publish not-ready worker addresses |
| C42 | Retry only transient HTTP execution-status read failures, with a bounded budget. Failed start/cancel/reset/CLI operations, run failures, and detected coordinator restarts remain terminal; no router-forwarding retries or analysis resubmission | User explicitly selected status-read retries as the sole exception to the earlier no-retry policy. Existing AMQP delivery semantics are not replaced; [Coordinator API](design.md#coordinator-api) |
| C43 | If an executor queue cannot accept a forwarded request, return the proposed terminal availability/overload error | User accepted error propagation for the exceptional overflow case; RabbitMQ enforces the queue bound. No extra Exaflow buffering or request-counting layer |
| C44 | Enhanced Flower participant-selection and round-completeness enforcement follows the initial integration | User deferred R19. Initial data/context/role isolation and Exaflow's C37 privacy checks remain required. No one-validation-worker restriction |
| C45 | Report per-worker AUCs and pooled log loss; no combined AUC | User selected local AUCs and pooled log loss. Pool log loss by evaluation row count; [Result](design.md#result) |
| C46 | The XGBoost outcome has exactly two categorical levels; `positive_class` maps to 1 and the other level to 0 | User selected exactly two categories rather than positive-versus-rest conversion; [Settings](design.md#settings) |
| C47 | Use `Recreate` for normal upgrades of the singleton controller and Flower coordinator Deployments | User accepted the rollout recommendation; [Images and deployment](design.md#images-and-deployment) |
| C48 | Coordinator admission counts pending, live, and uncertain work. At capacity, reject a new start with HTTP 429 / `flower_capacity_exceeded`, and propagate the error without retries | User accepted the defensive capacity guard; [Concurrency](design.md#concurrency) |
| C49 | Future preparation framework: explicit roles, fit/transform phases, fitted-state sharing, and execution cleanup; statistical choices belong to algorithm developers | Separate [PRD](prd/role-aware-preprocessing.md). Independent evaluation fits on training and transforms both training and validation; C50 excludes it from the initial work |
| C50 | Keep the initial integration simple: defer role-aware fitted preprocessing until after the initial implementation | User explicitly requested a separate future PRD. Reuse the current native preprocessing path now; track follow-ups and open items in [future-work.md](future-work.md) |

## Implementation recommendations

| ID | Recommendation | Details |
| --- | --- | --- |
| R06 | XGBoost defaults: 5 rounds, 1 local round, learning rate 0.1, `max_depth` 6 as the maximum, seed 0. Fixed: `base_score` 0.5, `min_child_weight = minimum_row_count / 4`, `max_bin` 32. Needs privacy sign-off | [Settings](design.md#settings) |
| R07 | Target Flower 1.43.0 and XGBoost 3.4.1 on Python 3.12 in the separate project. Resolve and test the lockfile during implementation | [References](design.md#references) |
| R09 | The reader deletes prepared files at the end of a run. The worker removes expired leftovers at the next preparation | [Prepared files](design.md#prepared-files) |
| R11 | Order of work in thin end-to-end steps | [Order of work](plan.md#order-of-work) |
| R12 | Containers start Flower's own `flower-superlink` and `flower-supernode` commands, without a wrapper | [How Exaflow uses Flower](design.md#how-exaflow-uses-flower) |
| R14 | After coordinator restart, cancel surviving backend executions before readiness. Signal lost attempts through instance identity and the existing status polling flow. The controller-startup reset is already confirmed by C40 | [Restart cleanup](design.md#restart-cleanup) |
| R16 | Use mandatory routing, publisher confirms, and returned-message handling for the router's forwarding publish. Queue rejection/error propagation is confirmed by C43, with no new forwarding retry under C42 | [Router failure cases](design.md#router-failure-cases) |
| R17 | For an initial deployment that fully trusts every permitted pod/operator, node authentication need not be a prerequisite. Consider omitting it while retaining TLS and enforced network isolation; it does not protect the shared Control API. This simplification is proposed, not a change to the existing C29 choice | [Security setup](design.md#security-setup) |
| R22 | Add expiry cleanup independent of future traffic and explicit shared file permissions; strengthen R09's next-preparation-only cleanup | [F05](design.md#f05-file-retention-and-permissions-priority-1-r22) |
| R23 | Finalize the manifest, encoding, role-disjointness, parameter-floor handoff, and XGBoost tree-aggregation contracts during implementation; verify matching producers/readers and output behavior before release | [F06](design.md#f06-a-concrete-data-and-model-contract-priority-1-r23) |

C42, C45, and C46 resolve the status-read retry, evaluation-output, and outcome
mapping questions. R23 still covers the implementation details of the manifest and
model exchange. See the [review follow-up](design.md#review-follow-up-2026-11-18).
The [future-work register](future-work.md) separates post-initial enhancements from
details and release gates that still need completing for the initial integration.

## Deferred enhancements

| ID | Enhancement | Current scope |
| --- | --- | --- |
| C49/C50 | Role-aware fit/transform preprocessing | Future work only; [FW01 and separate PRD](future-work.md#after-the-initial-implementation) |
| R19 | Strict participant selection and complete-round reply enforcement | C44 defers it until after initial Flower integration; [F02](design.md#f02-an-explicit-flower-strategy-adapter-priority-1-r19) records the runtime defaults |
| R25 | Independent native/Flower eligibility | Retain C41 and revisit if operational evidence warrants the added complexity; [F08](design.md#f08-whole-pod-exclusion-broadens-failure-impact-priority-2-r25) |

## Superseded

| Old proposal | Replaced by |
| --- | --- |
| C11: global-worker removal is out of scope | C19: it is a prerequisite |
| Validation only on the global worker | C07, then C19 |
| Original C07: exactly one validation worker | Revised C07: multiple validation workers are allowed |
| Intermediate C05/C27: fit/transform capabilities are part of the initial work | C50: unchanged native preprocessing initially; C49's framework is a separate future PRD |
| A gRPC coordinator API | C12 |
| Runtime specification hashes and version checks | C08, C09 |
| A durable SQLite registry and saved SuperLink state | C14 |
| Specification kept in the root algorithm tree, code elsewhere | C13 |
| A training-fit/replay preprocessing framework as an initial requirement | C50: defer it to the separate C49 PRD |
| Directory names `flower-runtime/`, `flower-runner/`, `flower-integration/` | C15 |
| Package names `flower`, `exaflow_flower`, `flower_runtime`, `flower_integration` | C18 |
| New worker gRPC operations to prepare and release data | C27 |
| R08: preparation UDF (recommendation) | C27 |
| R10: security setup with optional signed-app verification | C29, without signed apps |
| Flower limit as a second controller semaphore | C31, C34 |
| `POST /analysis` always on next to RabbitMQ, with in-process limits | C30, C34 |
| Separate `Exareme3Strategy` and `Exareme3WithAggregationServerStrategy` | C28 |
| Unknown algorithms forwarded to the `exareme3` queue for validation | C32 |
| Coordinator progress reporting, catalog, and release endpoints | C25 |
| SuperLink supervised inside the coordinator container | C25 |
| `max_depth` 8 | R06 |
| Skipping insufficient training workers/roles in a Flower request | C37: fail the requested analysis; let the caller explicitly choose a different dataset set |
| P2 temporarily executes the old Flower strategy | C38: explicit unavailable error until the replacement is enabled |
| R13: Fleet-only reverse proxy | C39: no extra proxy in the initial protected Kubernetes deployment |
| C24/R15: sidecar probe omission or separate eligibility with not-ready discovery | C41: whole-pod exclusion when either component is unready/unhealthy |
| Controller-startup reset as an unconfirmed part of R14 | C40: controller-startup reset is confirmed; coordinator restart details remain under R14 |
| Broad retries for Flower control/CLI operations or new router forwarding failures | Revised C42: only transient execution-status HTTP reads may retry |
| Original C42: no retries even for transient HTTP status-read failures | Revised C42: bounded retries for those reads; all other failure behavior stays terminal |
| A single combined AUC or weighted average of local AUCs | C45: per-worker AUCs and pooled log loss |
| Positive-class-versus-rest conversion for an outcome with more than two levels | C46: require exactly two levels |
| R24: move the runtime prototype before P1–P3 | C36 reaffirmed: the prerequisites simplify development and land first |
| R18: initially restrict the platform to fixed transformations | C05/C50: reuse current preprocessing now; offer general fit/transform capabilities later under C49 |
| R20: coordinator admission check as a recommendation | C48: confirmed admission guard with terminal capacity errors |
| R21: singleton `Recreate` rollout as a recommendation | C47: confirmed for normal controller/coordinator upgrades |
| Original C34: queue prefetch is the only Flower limit | C34/C48: delivery prefetch plus coordinator accounting for live work |
