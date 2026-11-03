# Controller modernization tasks

Tickets for [controller-modernization-plan.md](controller-modernization-plan.md).
Each ticket is one reviewable change with its own tests. Tickets use the Ornith
template, so a ticket body can go to `ornith_start(repo="exaflow", ticket_markdown=...)` unchanged. Confirm decisions D1 to D6 in the plan first.

Waves (tickets in one wave edit disjoint files and can run in parallel):

- Wave 1: T1, T2, T3, T4, T5, T7, T8.
- Wave 2: T6 (after T1, T2, T5), T9 (after T2), and T10.
- Wave 3, optional: C1, C2, C3, C4 in order.

## Review focus

Failure modes most likely to slip past a reviewer. The owning ticket has a test
for each one.

1. SIGTERM during an aggregation-backed analysis: cleanup still runs after the
   algorithm thread ends (T7 keeps `run_blocking` ordering; T2 stops the WLA
   first).
1. A worker fails in the middle of a round: sibling RPCs are cancelled, then
   aggregation cleanup releases waiting workers (T3, T4).
1. gRPC done-callbacks run on gRPC threads: they only enqueue an index and
   never raise or block (T3).
1. A late Flower result from a timed-out run arrives during the next run (T6).
1. A saturated pool with queued Flower requests: native analyses still get
   slots (T5).

______________________________________________________________________

## T1

Title: test(controller): pin the HTTP contract of every controller route

1. Files — Add `tests/standalone_tests/controller/test_http_contract.py`. No source changes.
1. Names — Drive `exaflow.controller.quart.app.app` through `app.test_client()`, as in `tests/standalone_tests/controller/test_threaded_endpoint_errors.py`. Stub `exaflow.controller.quart.endpoints.get_worker_landscape_aggregator`, `endpoints.get_flower_controller`, `endpoints.validate_analysis_request`, and `endpoints.get_algorithm_execution_strategy` with `monkeypatch`.
1. Keys — Routes: GET `/datasets`, `/datasets_locations`, `/datasets_variables`, `/cdes_metadata`, `/data_models_attributes`, `/specifications/inputdata`, `/specifications/preprocessing`, `/specifications/algorithms`, `/healthcheck`, `/livez`, `/flower/input`; POST `/wla`, `/analysis`, `/flower/result`.
1. Strings — Exact bodies: empty body for `/livez`, `/healthcheck`, `/wla`; JSON `{"message": "Result set successfully"}` for `/flower/result`; malformed `/analysis` body starts with `Analysis execution request malformed:`; `InsufficientDataError` returns `The algorithm could not run with the input provided because there are insufficient data.`; `AnalysisCapacityExceeded` returns `The controller is already running the maximum number of concurrent analyses.`; every other mapped error returns its `message` attribute.
1. Numbers — Status codes: 200, 400, 413, 460, 461, 462, 503, 512, 513.
1. Edge cases — `POST /analysis` with a body that fails `AnalysisRequestDTO` validation returns 400. Errors raised inside `asyncio.to_thread` keep their status (already covered by `test_threaded_endpoint_errors.py`; do not duplicate).
1. Behavior — No behavior change. For each route, assert the status code, the `Content-Type` header, and the body served today. Read the content types from a first run; do not guess them. Example: `GET /specifications/algorithms` returns 200 with body equal to `algorithm_specifications_dtos.model_dump_json()`. Raise each mapped exception from a stubbed `get_all_available_datasets_per_data_model` and call `GET /datasets`. For `POST /analysis` success, the stub strategy's `execute()` returns `'{"result": 1}'`, and the response body equals that string.
1. Unchanged — All files under `exaflow/`.
1. Verify — `uv run pytest -q tests/standalone_tests/controller`.
1. Acceptance criteria —
   - each metadata GET route returns 200 with JSON built from the WLA stub
   - each specifications route returns 200 with the DTO JSON body and its current content type
   - livez returns 200 with an empty body
   - analysis success returns the strategy JSON string with its current content type
   - malformed analysis body returns 400 with the malformed-request prefix
   - each mapped exception returns its mapped status and message (BadRequest 400, FilterError 400, BadUserInput 460, DataModelUnavailable 460, DatasetUnavailable 460, InsufficientDataError 461 with the fixed message, SMPCUsageError 462, WorkerMessageTooLargeError 413, AnalysisCapacityExceeded 503, WorkerUnresponsiveError 512, WorkerTaskTimeoutError 513)
   - flower input returns the registry inputdata as JSON
   - flower result returns 200 with the success message

______________________________________________________________________

## T2

Title: fix(controller): stop the worker landscape aggregator promptly and on shutdown

1. Files — `exaflow/controller/services/worker_landscape_aggregator/worker_landscape_aggregator.py`, `exaflow/controller/services/startup.py`, `exaflow/controller/quart/endpoints.py`, `exaflow/controller/services/exareme3/__init__.py`. Tests: new `tests/standalone_tests/controller/services/test_wla_lifecycle.py`.
1. Names — In `WorkerLandscapeAggregator`, replace the `_keep_updating` flag with `self._stop_event = threading.Event()`. `_update_loop` waits with `self._stop_event.wait(self._update_interval)` instead of `time.sleep(self._update_interval)` and exits when the event is set. `start()` clears the event; `stop()` sets it, then joins the thread. Add `stop_background_services() -> None` to `startup.py`; it calls `get_worker_landscape_aggregator().stop()`. In `endpoints.shutdown()`, run `await asyncio.to_thread(stop_background_services)` before the existing `shutdown_analysis_execution` call. `get_exareme3_controller()` raises when no controller was set, instead of building one.
1. Keys — Config `worker_landscape_aggregator_update_interval` (unchanged).
1. Strings — `get_exareme3_controller()` raises `ValueError("Exareme3Controller has not been initialized.")`, matching `get_flower_controller()`.
1. Numbers — With `update_interval=60`, `stop()` returns in under 1 s while the loop waits for the next refresh.
1. Edge cases — `stop()` before `start()` is a no-op. `start()` after `stop()` resumes refreshing. An exception from `update()` is logged and the loop continues (unchanged). A refresh in progress finishes before `stop()` returns.
1. Behavior — Before: shutdown leaves the daemon thread running, `stop()` blocks up to `update_interval` seconds, and `get_exareme3_controller()` can build a controller with a 0 s timeout from the missing config key `task_timeout`. After: shutdown stops the thread promptly, and an uninitialized getter fails loudly.
1. Unchanged — Refresh logic, `_update_lock`, the first refresh running immediately on `start()`, `start_background_services()`.
1. Verify — `uv run pytest -q tests/standalone_tests/controller`.
1. Acceptance criteria —
   - stop returns promptly while the loop waits for the next refresh
   - stop before start is a no-op
   - start after stop resumes refreshing
   - app shutdown stops the worker landscape aggregator before shutting down the analysis pool
   - get_exareme3_controller raises when no controller was set

______________________________________________________________________

## T3

Title: refactor(controller): fan out UDF calls with gRPC futures instead of a thread pool per call

1. Files — `exaflow/controller/worker_client/app.py`, `exaflow/controller/worker_client/tasks_handler.py`, `exaflow/controller/services/exareme3/tasks_handler.py`, `exaflow/controller/services/exareme3/algorithm_flow_engine_interface.py`. Tests: new `tests/standalone_tests/controller/services/test_udf_fanout.py`.
1. Names —
   - `WorkerClient.start_call(self, rpc_name: str, request: Any, timeout: int) -> grpc.Future` returns `getattr(self._stub, rpc_name).future(request, timeout=timeout)`.
   - `WorkerClient.finish_call(self, rpc_name: str, future: grpc.Future) -> Any` returns `future.result()`. On `grpc.RpcError` it logs the existing debug line and calls `self._handle_rpc_error(exc)`.
   - `WorkerClient.call` becomes `return self.finish_call(rpc_name, self.start_call(rpc_name, request, timeout))`, so there is one error path.
   - `WorkerTasksHandler.start_run_udf(self, request_id: str, udf_registry_key: str, kw_args: dict, system_args: dict, timeout: int) -> grpc.Future` and `WorkerTasksHandler.finish_run_udf(self, future: grpc.Future) -> dict` (result-format check and `decode_udf_result`). `WorkerTasksHandler.run_udf` stays, implemented as finish after start; tests in `test_worker_run_udf_response.py` and `test_worker_grpc_message_size.py` call it.
   - `Exareme3TasksHandler.start_run_udf(self, udf_registry_key: str, kw_args: dict, system_args: RunUdfSystemArgs) -> grpc.Future` and `Exareme3TasksHandler.finish_run_udf(self, future: grpc.Future) -> dict`. Remove `Exareme3TasksHandler.run_udf` once nothing calls it.
   - `Exareme3AlgorithmFlowEngineInterface.run_udf` keeps its signature and return type. It starts one future per handler, registers `future.add_done_callback(lambda _, idx=idx: done.put(idx))` on a `queue.SimpleQueue`, then takes `len(handlers)` indexes from the queue and calls `finish_run_udf` in completion order. Drop the `ThreadPoolExecutor` and `as_completed` imports.
1. Keys — n/a.
1. Strings — Keep exactly: warning `"Worker %s returned insufficient data and will be skipped: %s"` and error `"No workers had sufficient data to run the requested algorithm."`.
1. Numbers — Per-call deadline stays `tasks_timeout`. `run_udf` creates zero threads.
1. Edge cases — No handlers returns `[]`. One worker raising `InsufficientDataError` is skipped. Every worker raising it raises `InsufficientDataError`. Any other error cancels every future that is not done and raises the mapped error without waiting for the rest. If `start_run_udf` raises for handler k, the futures already started are cancelled and the error is raised.
1. Behavior — Before: each `run_udf` builds `ThreadPoolExecutor(max_workers=len(handlers))`; after a failure, sibling threads keep their RPCs open until the deadline. After: no executor, and sibling RPCs are cancelled on failure. Results keep handler order: handlers `[w1, w2]` that finish `w2` first still return `[r1, r2]`.
1. Unchanged — Error mapping to `WorkerClientTimeoutException`, `BadUserInput`, `InsufficientDataError`, `WorkerMessageTooLargeError`, `WorkerClientConnectionError`. Metadata, health, and Flower RPCs keep using `call`. Worker code, aggregation, protobufs.
1. Verify — `uv run pytest -q tests/standalone_tests/controller tests/standalone_tests/test_worker_run_udf_response.py tests/standalone_tests/test_worker_grpc_message_size.py tests/standalone_tests/test_udf_result_serialization.py tests/standalone_tests/test_grpc_recoverability.py`. Fake gRPC futures in tests with `concurrent.futures.Future`.
1. Acceptance criteria —
   - run_udf returns results in handler order when workers finish out of order
   - run_udf starts no threads
   - run_udf skips a worker that reports insufficient data
   - run_udf raises InsufficientDataError when every worker reports insufficient data
   - run_udf cancels pending sibling calls when one worker fails
   - run_udf cancels started calls when starting a later call fails
   - worker client call maps a DEADLINE_EXCEEDED future to WorkerClientTimeoutException

______________________________________________________________________

## T4

Title: fix(controller): put a deadline on aggregation Configure and Cleanup

1. Files — `exaflow/aggregation_clients/controller_aggregation_client.py`, `exaflow/controller/services/exareme3/strategies.py`. Tests: extend `tests/standalone_tests/controller/services/test_exareme3_aggregation_strategy.py`; update fakes in `tests/standalone_tests/controller/services/test_aggregation_cleanup_on_cancel.py` and `tests/standalone_tests/controller/test_blocking_calls_off_loop.py` if their signatures break.
1. Names — `ControllerAggregationClient.configure(self, num_workers: int, timeout: float) -> str` passes `timeout=timeout` to `self._stub.Configure`. `ControllerAggregationClient.cleanup(self, timeout: float) -> str` passes `timeout=timeout` to `self._stub.Cleanup`. `Exareme3WithAggregationServerStrategy.execute` passes `self._controller.task_timeout` to both.
1. Keys — Controller config `worker_tasks_timeout`, read through `Exareme3Controller.task_timeout`. No new key.
1. Strings — Keep the cleanup warning `"Aggregation cleanup for request_id=%s failed: %s %s"`.
1. Numbers — Deadline = `worker_tasks_timeout` seconds (decision D1; Helm default 300).
1. Edge cases — Configure exceeding its deadline raises `grpc.RpcError` as today; `configured` stays False, so Cleanup is not called (the aggregation server evicts contexts that never receive Cleanup). Cleanup exceeding its deadline returns `pb2.Status.ERROR` and logs the warning.
1. Behavior — Before: a hung aggregation server blocks the analysis thread and its slot forever. After: the call fails after `worker_tasks_timeout` seconds and the slot is released.
1. Unchanged — `ALREADY_CONFIGURED` handling, Cleanup only for a context this request configured, Cleanup only after `algorithm.run` finishes, worker-side aggregation clients.
1. Verify — `uv run pytest -q tests/standalone_tests/controller tests/standalone_tests/aggregation_server`.
1. Acceptance criteria —
   - configure sends the controller task timeout as the gRPC deadline
   - cleanup sends the controller task timeout as the gRPC deadline
   - cleanup past its deadline returns ERROR and logs a warning
   - configure past its deadline releases the analysis slot without calling cleanup

______________________________________________________________________

## T5

Title: fix(controller): reject a Flower analysis while another Flower analysis runs

1. Files — `exaflow/controller/services/flower/strategies.py`, `exaflow/controller/services/analysis_execution.py`. Tests: new `tests/standalone_tests/controller/services/test_flower_admission.py`.
1. Names — `AnalysisCapacityExceeded.__init__(self, message: str = <current default text>)`. In `FlowerStrategy.execute`, before `async with self._controller.algorithm_execution_lock:`, raise `AnalysisCapacityExceeded(FLOWER_BUSY_MESSAGE)` when `self._controller.algorithm_execution_lock.locked()`. There is no `await` between the check and the acquire, so the check cannot race. The existing 503 handler in `error_handlers.py` serves it.
1. Keys — n/a.
1. Strings — `FLOWER_BUSY_MESSAGE = "The controller is already running a Flower analysis."` (decision D2). The default capacity message stays unchanged.
1. Numbers — HTTP 503.
1. Edge cases — Lock free: runs as today. Lock held: 503 at once, with no garbage-collect or other worker call.
1. Behavior — Before: a second Flower request waits on the lock while it holds one of `max_concurrent_experiments` slots, so queued Flower requests can block every native analysis. After: it gets 503 at once and its slot is free.
1. Unchanged — One Flower execution at a time, Flower cleanup, native analysis admission.
1. Verify — `uv run pytest -q tests/standalone_tests/controller`.
1. Acceptance criteria —
   - flower analysis is rejected with 503 while another flower analysis runs
   - rejected flower analysis makes no worker calls
   - rejected flower analysis releases its analysis slot
   - flower analysis runs when no other flower analysis holds the lock

______________________________________________________________________

## T6

Title: fix(flower): accept Flower callbacks only from the running request

1. Files — `exaflow/controller/services/flower/flower_io_registry.py`, `exaflow/controller/services/flower/strategies.py`, `exaflow/controller/quart/endpoints.py`, `exaflow/algorithms/flower/inputdata_preprocessing.py`. Tests: new `tests/standalone_tests/controller/services/test_flower_io_registry.py`; header cases in `tests/standalone_tests/controller/test_http_contract.py` (from T1).
1. Names — `FlowerIORegistry.set_inputdata(self, inputdata: dict, request_id: str)` stores `self._request_id`; `_reset_sync` sets `self._request_id = None`. New `FlowerIORegistry.is_current(self, request_id: str | None) -> bool`. `FlowerStrategy.execute` passes `self._request_id`. `get_flower_input` and `set_flower_result` read header `X-Request-Id` and return 409 without touching state when `is_current(...)` is False. In the script, `REQUEST_ID = os.getenv("REQUEST_ID")` (the worker sets it in `exaflow/worker/flower/starter/starter_service.py`) and `X-Request-Id: REQUEST_ID` goes on GET `/flower/input` and on both POST `/flower/result` calls.
1. Keys — Header `X-Request-Id` (decision D3). Environment variable `REQUEST_ID` (existing).
1. Strings — 409 body: `Flower callback does not belong to the running analysis.` (decision D3).
1. Numbers — HTTP 409. Every `requests` call in the script gets `timeout=30` seconds (input, result, error result, `/cdes_metadata`).
1. Edge cases — No Flower execution running (`_request_id is None`): 409. Missing header: 409. Matching header: today's behavior. A result after `reset()`: 409, and the result is ignored. A reset plus a new request id while `await request.json` is in progress: 409, and the new execution's `result_ready` stays unset. `set_result(result, request_id)` re-checks `is_current` and does not await before `result_ready.set()`.
1. Behavior — Before: any POST to `/flower/result` completes the current execution, including a late post from a timed-out run. After: only a post that carries the running request id does. Running id `abc`: POST with `X-Request-Id: old` returns 409 and `result_ready` stays unset; POST with `X-Request-Id: abc` returns 200 `{"message": "Result set successfully"}`.
1. Unchanged — `set_result` status logic (an `"error"` key means FAILURE), valid-callback response bodies, `/cdes_metadata`, worker RPCs, protobufs.
1. Verify — `uv run pytest -q tests/standalone_tests/controller`. State the callback contract change in the PR body.
1. Acceptance criteria —
   - result with the running request id completes the execution
   - result with another request id returns 409 and the execution keeps waiting
   - result without the header returns 409
   - callback while no flower execution runs returns 409
   - input with the running request id returns the inputdata
   - flower script sends the request id header and a timeout on every controller call
   - a result whose body is read after the running request changes returns 409 and does not complete the new execution

______________________________________________________________________

## T7

Title: fix(controller): let Kubernetes stop the controller gracefully

1. Files — `exaflow/controller/Dockerfile`, `exaflow/controller/quart/hypercorn_config.py`, `kubernetes/templates/exaflow-controller.yaml`. Tests: extend `tests/standalone_tests/controller/test_controller_probes.py`.
1. Names — Dockerfile: `CMD ["python", "-m", "hypercorn", "--config", "python:exaflow.controller.quart.hypercorn_config", "-b", "0.0.0.0:5000", "exaflow/controller/quart/app:app"]`. `hypercorn_config.py`: `workers = 0`, `graceful_timeout = 120`, `shutdown_timeout = 60`, `shutdown_rpc_cancel_timeout = 5`, `shutdown_cleanup_timeout = 10`, `shutdown_grace_margin = 20`. Template: `terminationGracePeriodSeconds` is the sum of those five values.
1. Keys — Hypercorn config `workers`, `graceful_timeout`, `shutdown_timeout`, `shutdown_rpc_cancel_timeout`, `shutdown_cleanup_timeout`, `shutdown_grace_margin`. The same five numbers are Helm values under `controller`. Pod spec `terminationGracePeriodSeconds`.
1. Strings — n/a.
1. Numbers — `graceful_timeout = 120` (Hypercorn default 3). `shutdown_timeout = 60`. `shutdown_rpc_cancel_timeout = 5`. `shutdown_cleanup_timeout = 10`. `shutdown_grace_margin = 20`. `terminationGracePeriodSeconds` = 215 at the chart defaults (Kubernetes default grace is 30). Decision D4, corrected: a call that starts near the end of the drain is cancelled with its worker RPC instead of running out `worker_tasks_timeout` (which would finish near t=420) and then a second cleanup of the same length. `after_serving` has `shutdown_timeout`.
1. Edge cases — No request in flight: the process exits right after SIGTERM. An analysis still running when the drain ends has its worker and aggregation RPCs cancelled; `run_blocking` waits at most `shutdown_rpc_cancel_timeout`, then aggregation cleanup uses `shutdown_cleanup_timeout`. A client disconnect while the process is not shutting down still waits for the analysis thread. WLA `stop()` sets the stop event even if the thread is not alive yet, and `start()` still joins without a timeout; shutdown passes `join_timeout=shutdown_cleanup_timeout`. `/livez` fails once the listen socket closes. On Kubernetes 1.25 and newer, `SyncTerminatingPod` calls `StopLivenessAndStartup` before `killPod` (see `kubelet.go` in v1.25.0 and v1.32.0), so liveness does not restart the container for that failure. The probe does not set its own `terminationGracePeriodSeconds`.
1. Behavior — Before: `sh -c` is PID 1 and does not forward SIGTERM (dash 0.5.12 keeps the `sh` process for `sh -c "<cmd>"`), so Kubernetes sends SIGKILL after 30 s and `after_serving` never runs. Sizing grace as `worker_tasks_timeout + 200` still waits out a worker RPC that shutdown can cancel. After: Hypercorn is the serving process, receives SIGTERM, drains for up to 120 s, cancels leftover RPCs, runs a short cleanup and the shutdown hooks, and the pod stays up for that budget.
1. Unchanged — Bind address and port 5000, access-log settings, probe paths and thresholds, image user, `PYTHONPATH`, per-call `worker_tasks_timeout` on calls that are not being shut down.
1. Verify — `uv run pytest -q tests/standalone_tests/controller/test_controller_probes.py tests/standalone_tests/controller/services/test_shutdown_budget.py` and `helm template kubernetes/`. `kubernetes/AGENTS.md` requires a rendered chart and human review for deployment changes.
1. Acceptance criteria —
   - controller image command is exec form without a shell
   - hypercorn graceful timeout is 120 seconds and shutdown timeout is 60 seconds
   - hypercorn serves in-process so the app observes SIGTERM
   - rendered controller pod grace period equals graceful timeout plus rpc-cancel bound plus cleanup bound plus shutdown timeout plus 20 seconds
   - at the chart default that grace period is 215 seconds
   - overriding the cleanup bound changes the rendered grace; overriding worker_tasks_timeout does not
   - a shutdown cancel aborts an in-flight RPC instead of waiting out worker_tasks_timeout
   - a client disconnect still waits for the analysis thread

______________________________________________________________________

## T8

Title: chore(tasks): stop the local controller with SIGTERM before SIGKILL

1. Files — `tasks.py` (`kill_controller`).
1. Names — New helper `_terminate_pids(c, pids: str, timeout: float | None = None) -> None`: run `kill <pids>`, poll `kill -0` every 0.5 s until `timeout` (default the T7 shutdown budget), then `kill -9` the pids still alive. Both the `lsof` path and the `pgrep` fallback of `kill_controller` call it instead of `kill -9 {pids} && sleep 1`. `kill_controller` takes `stop_timeout`.
1. Keys — Invoke flag `--stop-timeout`. Environment variable `CONTROLLER_STOP_TIMEOUT`. The flag wins when both are set.
1. Strings — Replace `Killing previous Hypercorn instances...` with `Stopping previous Hypercorn instances...`. Keep `No hypercorn instance found` and the non-controller listener warning.
1. Numbers — Default maximum wait is `graceful_timeout + shutdown_rpc_cancel_timeout + shutdown_cleanup_timeout + shutdown_timeout` (195 s at the T7 defaults). Poll interval 0.5 s (decision D5, corrected: 10 s SIGKILL interrupted the 120 s drain).
1. Edge cases — Process already gone: no error, and the wait returns immediately. Process ignoring SIGTERM: SIGKILL after the timeout. `--stop-timeout 1` or `CONTROLLER_STOP_TIMEOUT=1` SIGKILLs after 1 s. Port held by a non-controller process: warning only (unchanged).
1. Behavior — Before: `kill -9` skips Hypercorn's shutdown hooks. A 10 s cap still SIGKILLs during the drain. After: SIGTERM comes first and the default wait covers the drain and the shutdown hooks, so `after_serving` can finish locally as it does in Kubernetes after T7. A local restart can pass a shorter timeout.
1. Unchanged — Process matching (`python` or `hypercorn` listening on the controller port; the two `pgrep -f` patterns), worker and aggregation-server kill tasks.
1. Verify — `uv run inv start-controller --detached`, then `uv run inv kill-controller`. The controller log shows Hypercorn shutdown lines and the port is free.
1. Acceptance criteria —
   - kill-controller sends SIGTERM before SIGKILL
   - kill-controller returns once the controller exits on SIGTERM
   - kill-controller sends SIGKILL to a controller still alive after the shutdown budget
   - `--stop-timeout` and `CONTROLLER_STOP_TIMEOUT` shorten that wait

______________________________________________________________________

## T9

Title: fix(controller): select an analysis's workers from one WLA snapshot

1. Files — `exaflow/controller/services/worker_landscape_aggregator/worker_landscape_aggregator.py`, `exaflow/controller/services/controller_interface.py`. Tests: extend `tests/standalone_tests/controller/services/test_worker_landscape_aggregator.py`.
1. Names — New `WorkerLandscapeAggregator.get_workers_info_with_any_of_datasets(self, data_model: str, datasets: List[str]) -> List[WorkerInfo]`. It reads `registries = self._registries` once, gets ids from `registries.data_model_registry.get_worker_ids_with_any_of_datasets(data_model, datasets)`, and maps them through `registries.worker_registry.workers_per_id`. `ControllerI.get_local_worker_tasks_handlers` calls it instead of `get_worker_ids_with_any_of_datasets` followed by one `get_worker_info` per id.
1. Keys — n/a.
1. Strings — n/a.
1. Numbers — One `_registries` read per call.
1. Edge cases — No worker holds the datasets: empty list. A refresh that replaces `_registries` during the call does not change the result of that call.
1. Behavior — Before: a refresh between the id lookup and `get_worker_info` can raise `KeyError`, which surfaces as HTTP 500. After: ids and worker info come from the same snapshot.
1. Unchanged — `get_worker_ids_with_any_of_datasets` and `get_worker_info` stay for other callers. Validation and refresh logic.
1. Verify — `uv run pytest -q tests/standalone_tests/controller`.
1. Acceptance criteria —
   - returns worker info for every worker holding any requested dataset
   - returns an empty list when no worker holds the datasets
   - result stays consistent when registries are replaced after the first read
   - local worker task handlers are built from one snapshot

______________________________________________________________________

## T10

Title: fix(worker): deliver SIGTERM to the worker process

1. Files — `exaflow/worker/Dockerfile`, `exaflow/worker/bootstrap.sh`, `exaflow/worker/grpc_server.py`. Tests: `tests/standalone_tests/test_worker_image.py`, `tests/standalone_tests/test_worker_sigterm.py`.
1. Names — Dockerfile `CMD ["/home/bootstrap.sh"]`. `bootstrap.sh` runs `exec python -m exaflow.worker.grpc_server`. `serve` installs SIGTERM and SIGINT handlers that call `server.stop(SHUTDOWN_GRACE_SECONDS)` before `wait_for_termination`.
1. Keys — n/a.
1. Strings — n/a.
1. Numbers — `SHUTDOWN_GRACE_SECONDS = 10`. The worker chart does not set `terminationGracePeriodSeconds`, so this fits Kubernetes' default 30 s grace.
1. Edge cases — The script is PID 1 through its shebang. `exec` replaces that shell with the gRPC server, so SIGTERM reaches the server. There is no extra `sh -c` shell. Python as PID 1 still ignores SIGTERM until `serve` installs a handler; the handler is installed after `start` and before the ready log. SIGINT uses the same grace.
1. Behavior — Before: `CMD ["sh", "-c", "/home/bootstrap.sh"]` leaves a shell as PID 1, and `bootstrap.sh` starts the server without `exec`, so SIGTERM never reaches the worker. Even with exec form, `wait_for_termination` installs no SIGTERM handler, so container PID 1 stays up until SIGKILL. After: the command is exec form, the script replaces itself with the server, and the server stops within the grace period.
1. Unchanged — Image user, bootstrap path `/home/bootstrap.sh`, server module, environment, and the copied codebase.
1. Verify — `uv run pytest -q tests/standalone_tests/test_worker_image.py tests/standalone_tests/test_worker_sigterm.py`.
1. Acceptance criteria —
   - worker image command is exec form without a shell
   - bootstrap.sh execs the gRPC server
   - a worker process that starts with SIGTERM ignored still exits when SIGTERM arrives

______________________________________________________________________

## Phase C: FastAPI and Uvicorn

D6 is yes. `/docs`, `/redoc`, and `/openapi.json` stay disabled. Gate: the T1
contract tests pass unchanged against the new app; only the client fixture
(the app import) changes. C1, C2, C3, and C4 are done. Uvicorn `--workers 1`
is in-process. `--timeout-graceful-shutdown 120` is the drain. The 60 s hook
budget stays in the pod grace because Uvicorn has no separate hook timeout.

- C1 `refactor(controller): table-driven error responses`. In `error_handlers.py`, replace the eleven handler functions with one mapping from exception type to (status, log level) and register one handler per entry. `InsufficientDataError` keeps its fixed message. Delete the commented-out generic handler and the unused `HTTPStatusCode.UNEXPECTED_ERROR` member. Pure refactor; T1 is the regression net.
- C2 `refactor(controller): move analysis orchestration out of the route`. New `async def run_analysis(request_body: Any) -> str` in `exaflow/controller/services/analysis_service.py` holds DTO parsing, `validate_analysis_request`, strategy creation, `analysis_slot()`, and `strategy.execute()`. The `/analysis` route becomes one call.
- C3 `feat(controller): serve the controller API with FastAPI`. Add `exaflow/controller/http/app.py` with a lifespan that calls `start_background_services`, `start_analysis_execution`, and the T2 shutdown order. Return `PlainTextResponse` wherever Quart returned a `str` today, so bodies are not JSON-quoted. Read `/analysis` with `await request.json()` and no Pydantic body parameter, so FastAPI never answers 422. Register C1's table as exception handlers. Keep Hypercorn. Add `fastapi` to the `controller` group and `httpx` to `dev`, run `uv lock`, delete `exaflow/controller/quart/` except `hypercorn_config.py` and `loggers.py`, and update `AGENTS.md`, `architecture.md`, and `module-index.md`.
- C4 `chore(controller): run the controller with Uvicorn`. Done. Exec form: `uvicorn exaflow.controller.http.app:app --host 0.0.0.0 --port 5000 --workers 1 --timeout-graceful-shutdown 120 --log-config exaflow/controller/http/log_config.json`. `--workers 1` is in-process. Startup `dictConfig` still applies the federation-aware format from `loggers.py` after Uvicorn loads the JSON file. `hypercorn` and `hypercorn_config.py` are removed. The budget constants live in `exaflow/controller/http/server_config.py`.
