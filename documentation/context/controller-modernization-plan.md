# Controller modernization plan

Status: In progress. D1, D2, and D5 confirmed as proposed. D4 revised:
`terminationGracePeriodSeconds` is `worker_tasks_timeout` + 200 s (500 s at
the Helm default of 300 s) so an in-flight worker call finishes inside the
pod. Wave 1 (T1, T2, T3, T4, T5, T7, T8) is
[madgik/exaflow#663](https://github.com/madgik/exaflow/pull/663); not merged.
Wave 2 (T6, T9, T10, and the D4 revision) is implemented on
`fix/controller-modernization-wave2`, stacked on #663; not merged.
Updated: 2026-11-03. Revises commit `ba2978cd` ("Initial
plan", branch `fix/controller-off-event-loop-and-liveness`), reviewed against
`master`. Implementation tickets:
[controller-modernization-tasks.md](controller-modernization-tasks.md).

## Summary

The initial plan proposed eight stages: baselines, a shared runtime object,
`grpc.aio` behind a thread-to-loop bridge, WLA and Flower on the event loop,
Quart to FastAPI, Hypercorn to Uvicorn, RabbitMQ with a SQLite job ledger and
outbox, and a staged rollout.

Review against the code shows that the defects worth fixing are small and
independent of the web framework, while the framework, server, and broker
changes carry most of the cost and little of the benefit today. The revised
plan therefore:

1. Pins the HTTP contract with tests before anything touches the HTTP layer.
1. Fixes concrete robustness defects: unbounded aggregation RPCs, per-call
   thread pools that cannot cancel, a WLA thread that never stops, Flower
   requests that hold analysis slots, uncorrelated Flower callbacks, and a
   container that never receives SIGTERM.
1. Keeps Quart and Hypercorn. FastAPI and Uvicorn become an optional later
   phase that is cheap once the HTTP layer is thin.
1. Drops RabbitMQ, the job ledger, and the outbox until a consumer exists.

## Review of each proposed change

| Initial plan item | Verdict | Evidence | Replacement |
| --- | --- | --- | --- |
| 0. HTTP contract capture | Keep | Only `/livez` and two error paths have endpoint tests. The backend parses bodies and status codes. | T1 |
| 0. Performance baseline at 1, 8, N, >N | Drop as a stage | No revised task claims a speed gain. `tests/prod_env_tests` already compares concurrent and serial results. | Measure only in a task that claims speed. |
| 1. `ControllerRuntime` and `AnalysisService` | Shrink | Several module-level singletons exist, but there is no second transport. The real lifecycle defects are local: the WLA thread is never stopped, and `get_exareme3_controller()` lazily builds a controller from the missing key `task_timeout`, which gives a 0 s gRPC timeout. | T2 now. Service extraction moves to optional C2. A runtime object waits for a second entry point. |
| 1. One pinned WLA snapshot per request | Keep, narrow | `ControllerI.get_local_worker_tasks_handlers` reads the registries twice. A refresh in between can raise `KeyError` (HTTP 500). | T9 (low priority) |
| 2. `grpc.aio` with `run_coroutine_threadsafe` bridge | Replace | `algorithm.run()` stays synchronous in a pool thread, so loop-owned channels add a bridge, loop-ownership rules, and deadlock risk for no gain. Synchronous gRPC already has `.future()`: one thread can start every worker call, wait for completions, and cancel siblings. This removes the per-call `ThreadPoolExecutor` without a bridge. | T3 |
| 2. Finite deadlines on aggregation Configure and Cleanup | Keep | `ControllerAggregationClient.configure()` and `.cleanup()` pass no timeout. A hung aggregation server blocks an analysis thread and its slot forever. | T4 |
| 2. Async DNS and channel reuse for aggregation | Drop | DNS lookup and channel creation run on the analysis thread, not the loop, and cost milliseconds per analysis. | None |
| 3. WLA daemon thread to asyncio task, async fan-out | Replace | Needs async clients for no benefit. The real defect: `stop()` joins a thread that sleeps `update_interval` seconds, and shutdown never calls it. The last good snapshot is already kept when a refresh fails. | T2 (event-based stop, stop on shutdown) |
| 3. Flower requests occupying analysis slots | Keep, simpler | `FlowerStrategy.execute` waits for `algorithm_execution_lock` inside `analysis_slot()`. Queued Flower requests can hold every slot. | T5: reject with 503 while busy |
| 3. Flower callback correlation | Keep, simpler | Workers already export `REQUEST_ID` to Flower processes (`exaflow/worker/flower/starter/starter_service.py`), so no protobuf change is needed. Callback `requests` calls have no timeout. | T6 |
| 3. Async Flower worker RPCs | Drop | `run_blocking` already keeps these calls off the loop. | None |
| 4. Quart to FastAPI | Defer, optional | No runtime gain. Adds `fastapi`, `starlette`, and `httpx` (none are in `uv.lock`). Defaults put the 400 and plain-text contracts at risk (422 validation, JSON-quoted strings). | Optional phase C |
| 5. Hypercorn to Uvicorn | Defer with phase C; take the real fixes now | The image runs `CMD ["sh", "-c", ...]`. With dash 0.5.12, `sh -c "<cmd>"` stays the parent process, so as PID 1 it never forwards SIGTERM: Kubernetes kills the pod after the default 30 s and shutdown hooks never run. Hypercorn `graceful_timeout` defaults to 3 s. The chart sets no `terminationGracePeriodSeconds`. `tasks.py` stops the local controller with `kill -9`. | T7, T8 |
| 6. RabbitMQ, SQLite job ledger, outbox | Drop | `platform-backend` calls `POST /analysis` synchronously (`HTTPUtil.sendPost`), and no broker exists in the `mip` deployment. The stage adds a persistent volume, reconciliation, and dead-letter operations for a consumer that does not exist. | Revisit when the backend commits to async submission. Start then with an HTTP job API on the C2 service function. |
| 7. Rollout: `Recreate`, PVC, broker settings | Drop | Needed only for durable MQ state. With in-memory state, `RollingUpdate` keeps zero-downtime deploys. | Grace period in T7 |

## Revised roadmap

Phase A, safety net: T1 pins the HTTP contract.

Phase B, robustness:

| Ticket | Outcome | Priority |
| --- | --- | --- |
| T2 | WLA stops promptly and on shutdown; no broken lazy controller getter | P1 |
| T3 | UDF fan-out through gRPC futures; no thread pool per call | P1 |
| T4 | Deadline on aggregation Configure and Cleanup | P1 |
| T5 | Flower request rejected while another Flower analysis runs | P2 |
| T6 | Flower callbacks accepted only from the running request | P2 |
| T7 | Container receives SIGTERM and drains before exit | P1 |
| T8 | Local `kill-controller` sends SIGTERM before SIGKILL | P3 |
| T9 | Worker selection reads one WLA snapshot | P3 |

Phase C, optional: C1 table-driven error responses, C2 analysis service
function, C3 FastAPI adapter served by Hypercorn, C4 Uvicorn. Gate: the T1
contract tests pass unchanged except for the client fixture.

Execution waves (tickets in one wave edit disjoint files):

| Wave | Tickets | Depends on |
| --- | --- | --- |
| 1 | T1, T2, T3, T4, T5, T7, T8 | None |
| 2 | T6, T9, T10 | T6: T1, T2, T5. T9: T2. T10: none. |
| 3 (optional) | C1, then C2, then C3, then C4 | Wave 2 |

## Decisions needed before delegation

The tickets use these values. None of them is in the code today; confirm or
change them first.

| ID | Ticket | Proposed value |
| --- | --- | --- |
| D1 | T4 | Configure and Cleanup deadline = `worker_tasks_timeout` (Helm default 300 s). No new config key. |
| D2 | T5 | 503 message: "The controller is already running a Flower analysis." |
| D3 | T6 | Header `X-Request-Id`. Mismatch returns 409 with "Flower callback does not belong to the running analysis." Callback HTTP timeout 30 s. |
| D4 | T7 | Hypercorn `graceful_timeout = 120` s and `shutdown_timeout = 60` s. `terminationGracePeriodSeconds` = `worker_tasks_timeout` + 200 (120 s drain + 60 s shutdown + 20 s margin; 500 s at the Helm default of 300 s). The per-call worker timeout stays. The pod grace period is sized from it so a call that starts near the end of the drain can finish before SIGKILL. |
| D5 | T8 | Wait 10 s after SIGTERM before SIGKILL. |
| D6 | C | Whether to run phase C at all. If yes, whether FastAPI's `/docs`, `/redoc`, and `/openapi.json` routes stay enabled. |

## Invariants kept from the initial plan

- One controller process with one web worker per federation. In-memory state
  and locks are process-local.
- `POST /analysis` stays synchronous and returns 503 at capacity.
- An analysis thread owns its lifetime: cleanup waits for the thread
  (`run_blocking`), also on client disconnect.
- Per-call worker timeouts stay. There is no overall analysis deadline.
- Worker and aggregation protocols, protobufs, and algorithm code stay
  unchanged.

## Compatibility and rollback

- T6 changes the Flower callback contract: `/flower/input` and
  `/flower/result` require `X-Request-Id`. The bundled Flower scripts change
  in the same commit. Roll back the controller and scripts together.
- T7 changes the controller image command and the Helm pod grace period.
  The grace period is `worker_tasks_timeout + 200` seconds (500 at the
  default). Render the chart and get the deployment review required by
  `kubernetes/AGENTS.md`. Roll back with the previous image and chart.
- T10 changes the worker image command. Roll back with the previous image.
- Other tickets change no public route, request or response shape, protobuf,
  or Helm value. Roll back by reverting the commit.

## Follow-ups outside this plan

- Worker SIGTERM delivery is T10. The worker command is exec form and
  `bootstrap.sh` replaces itself with the gRPC server.

## Validation

```bash
uv run pytest -q tests/standalone_tests/controller
uv run pytest -q tests/standalone_tests/aggregation_server tests/standalone_tests/test_grpc_recoverability.py tests/standalone_tests/test_worker_grpc_message_size.py tests/standalone_tests/test_udf_result_serialization.py tests/standalone_tests/test_worker_run_udf_response.py
helm template kubernetes/
```

## References

- [Current architecture and boundaries](architecture.md).
- [Existing analysis lifecycle decision](decision-log.md#decision-bound-controller-analyses-and-keep-liveness-off-that-pool).
- [gRPC Python futures](https://grpc.github.io/grpc/python/grpc.html#grpc.UnaryUnaryMultiCallable.future).
