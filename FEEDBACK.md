# Exaflow review: general feedback

Review of `master` at `e8c15aaf3` (2026-10-12). Findings that were fixed live on
three companion branches; everything else is listed here as recommendations,
ordered by impact.

## Companion branches

All three branches start from `master` and pass the standalone suite on their
own. Merged together with this branch (perf, bugfixes, enhancements, feedback,
in that order) they merge without conflicts and pass all 1544 standalone tests.
Environment-backed suites (`tests/algorithm_validation_tests`,
`tests/prod_env_tests`) were not run because they need a deployed federation.

| Branch | Commits | Contents |
|---|---|---|
| `claude/perf-speedups` | 4 | Fewer aggregation-server round-trips, cached lazy-aggregation rewrite, faster worker startup. |
| `claude/bugfixes` | 14 | SQL injection, controller availability, resource leaks, validation gaps, statistics errors. |
| `claude/algorithm-enhancements` | 8 | Effect sizes, Welch's ANOVA, odds ratios, t-test summaries. |

### `claude/perf-speedups`

- Fuse independent SUM aggregations into one aggregation-server call through
  the new `fused_sum` and `fused_min_max` helpers. Each call is a barrier across
  all workers, so the number of calls dominates latency. Round-trips before and
  after:
  - logistic regression: 3 -> 1 per Newton iteration;
  - OLS: 7 -> 2 (also used by each linear-regression CV fold);
  - KMeans: 2 -> 1 per Lloyd iteration;
  - Pearson: 6 -> 1; one-way ANOVA: 9 -> 2;
  - describe: 10 + one per nominal variable -> 2 per dataset;
  - t-tests: one-sample 5 -> 1, paired 3 -> 1, independent 2 -> 1;
  - Gaussian Naive Bayes: 3 -> 1;
  - CV scorers: classification 8 -> 1, regression 5 -> 1 per fold.
    Results are numerically identical.
- Cache the lazy-aggregation UDF rewrite per UDF; it was re-parsed and
  re-compiled on every `RunUdf` call (~1.3 ms each).
- Worker startup reads each CSV once instead of four times and inserts CDE
  metadata in one batch: 8.0 s -> 3.8 s on the dementia fixture, with a
  byte-identical DuckDB database.
- KMeans local sums use `bincount` instead of `np.add.at` (~2x faster, same
  summation order).

### `claude/bugfixes`

- **Security:** filter values were quoted according to the client-supplied rule
  `type`, so a string sent with `"type": "integer"` passed validation and was
  spliced into the worker's DuckDB query as raw SQL. Values are now quoted from
  their Python type and identifiers are escaped. Worker metadata queries now use
  bound parameters.
- **Controller availability:** `algorithm.run()` executed on the Quart event loop
  of the single Hypercorn worker, blocking every other request for the whole
  analysis; analyses, `/healthcheck` and `/wla` now run in threads. The liveness
  probe called `/healthcheck`, which fails when any single worker is down, so one
  bad worker made the kubelet restart a healthy controller; liveness now uses a
  new shallow `GET /livez` (Helm template change, startup probe unchanged).
- **Resource leaks:** the worker never closed the aggregation gRPC channel on
  successful UDF runs, and the aggregation server left workers blocked for up to
  `max_wait_for_aggregation_inputs` (120 s in prod values) after `Cleanup`.
- **Transport:** the worker `RunUdf` path used gRPC's 4 MiB default message
  limit while the aggregation server allows 100 MiB; both sides now use 100 MiB.
- **Error masking:** the lazy-aggregation executor swallowed batch failures and
  replayed them as individual calls on new aggregation steps.
- **Validation:** falsy parameter values (`0`, `0.0`) skipped type, enum and
  min/max checks; booleans were accepted as ints; empty dataset lists and a
  missing global worker produced HTTP 500 instead of a clear error.
- **Statistics:** one-sided t-test confidence bounds used the two-sided critical
  value; t-test and ANOVA p-values underflowed to 0 (`1 - cdf`); the independent
  t-test returned NaN with a single-observation group; one-way ANOVA reported
  group "sample" standard deviations scaled by `(n-1)/n` instead of
  `n/(n-1)`; Pearson returned NaN or rounding noise for constant columns instead
  of the documented 0.
- **Flower data filters:** `OR` dropped genuinely identical rows, `not_in` was
  unsupported, and filter-only columns were never read from the CSV.

### `claude/algorithm-enhancements`

- Chi-squared: Cramer's V.
- One-way ANOVA: eta-squared, omega-squared, and Welch's heteroscedastic ANOVA
  (matches `statsmodels.stats.oneway.anova_oneway(use_var="unequal")`).
- Two-way ANOVA: partial eta-squared per term.
- Logistic regression: odds ratios with confidence bounds (mirrors the hazard
  ratios of Cox regression).
- Independent t-test: group sizes and means (already promised by the
  specification) and Hedges' g. Paired t-test: `n_obs`.
- Doc fixes found on the way: the one-way ANOVA page described output fields that
  do not exist, and the chi-squared page named the p-value field `p`.

All new result fields are additive. The bugfix branch changes some values
(one-sided t-test bounds, tiny p-values, ANOVA group standard deviations,
Pearson for constant columns) and rejects requests that relied on the
validation gaps.

## Recommendations not implemented

### High

- **H1. UDF arguments travel as `google.protobuf.Struct`**
  (`exaflow/protos/worker/worker.proto:161-162`). Struct stores every number as a
  double, so ints arrive as floats on the worker (`3` -> `3.0`), integers above
  2^53 lose precision, and large arrays are slow to encode. The code base works
  around it with defensive `int(...)` casts in many UDFs. Results already use
  JSON bytes (`json-bytes-v1`); doing the same for `kw_args` and `system_args`
  would remove the casts. This is a protobuf contract change.
- **H2. Lazy-aggregation rewrite executes into the UDF module's globals**
  (`exaflow/worker/exareme3/lazy_aggregation/lazy_aggregation.py:156-160`) and
  temporarily rebinds the function name. Two concurrent rewrites of the same UDF
  can leave the rewritten function bound in the module. The perf branch caches
  the rewrite so it happens once per process; building the function with
  `types.FunctionType(code, func.__globals__, ...)` instead of `exec` into the
  module would remove the race. With concurrent analyses (bugfix branch) this
  matters more.

### Medium

- **M1. `/datasets_variables` is always empty.** The loader inserts `NULL`
  dataset properties (`exaflow/worker/utils/duck_db_csv_loader.py`, datasets
  insert), so `DatasetInfo.variables` is `[]` for every dataset.
  platform-backend fetches and forwards this map. Either populate it (columns
  with at least one non-null value per dataset) or remove the endpoint.
- **M2. Data model tags are never populated.** `is_longitudinal`
  (`worker_landscape_aggregator.py:118`) looks for a `longitudinal` tag, but the
  loader stores only `{"cdes": metadata}` and the fixture metadata uses
  `"longitudinal": true`. `is_longitudinal` has no callers today; wire it up or
  delete it.
- **M3. `_ensure_workers_data_loaded`** (`worker_landscape_aggregator.py:601`)
  requests the dataset list twice with nothing in between, and marks a worker as
  loaded even when both calls fail, so it is never retried. It looks like a
  leftover from a removed "load data" RPC.
- **M4. Worker landscape updates are sequential.** `_get_workers_info` and
  `_get_data_models_metadata_per_worker` (`worker_landscape_aggregator.py:658`)
  make 1 + 2 x data-models RPCs per worker, one after another; a single slow
  worker delays the registry refresh for the whole federation. A
  `ThreadPoolExecutor` per update would bound it by the slowest worker.
- **M5. New DuckDB connection per UDF call**
  (`exaflow/worker/exareme3/udf/udf_db.py:74`). Opening the read-only connection
  takes ~13 ms of the ~16 ms data load on the test fixture. A process-wide
  read-only connection with per-call cursors would remove it; it must be opened
  after the startup loader closes its read-write connection.
- **M6. Raw-moment variance formulas.** Pearson, describe and the t-tests compute
  variance as `n * sum(x^2) - sum(x)^2`, which loses all precision when the mean
  is large relative to the spread (e.g. values around 1e8 with standard
  deviation 1). A two-pass computation (global mean, then centered sums) costs
  one more aggregation round-trip and is exact enough for any realistic data.
- **M7. One-sample t-test `mean_diff` holds the sample mean**, not `mean - mu`
  (`exaflow/algorithms/federated/statistics/ttest_onesample.py:73`). The
  validation fixtures encode the same convention, so renaming needs a
  coordinated UI change.
- **M8. Readiness/startup semantics.** The startup probe still uses
  `/healthcheck`, so a controller started while a worker is down keeps
  restarting until every worker answers. Decide whether a partial federation is
  acceptable for serving and, if so, check only the global worker or require a
  quorum.
- **M9. `anova_twoway` `sstype`** is declared `INT` but its default is the
  string `"2"` (`exaflow/algorithms/exareme3/statistics/anova_twoway.py:78`).

### Low

- **L1. Dead code:** `_dataset_codes_from_metadata` (`duck_db_csv_loader.py:269`),
  `WorkerTaskResult` (`exaflow/controller/worker_client/tasks_handler.py:99`).
- **L2.** `convert_csv_paths_to_absolute` (`worker_info_db.py:125-126`) produces
  paths with `//`; `pathlib` joins would avoid it.
- **L3.** `worker_info_db.py:17` names its logger `"WORKER DATA LOADER"`
  (copy-paste).
- **L4.** `WorkerClientFactory` (`exaflow/controller/worker_client/app.py:76`)
  caches a channel per address forever; pod IP churn accumulates channels.
- **L5.** 15 files are committed with CRLF line endings (for example
  `exaflow/algorithms/federated/statistics/chi_squared.py`,
  `numpy_aggregator.py`, several tests). A `.gitattributes` with
  `* text=auto eol=lf` plus `git add --renormalize .` would stop editors from
  producing whole-file diffs.
- **L6.** The GLMM ordinal parity tests take 10-25 s each and dominate the
  standalone run; marking them `slow` would keep the default loop fast.

## How the review was done

- Read the runtime path end to end: HTTP endpoint, validation, strategy, flow
  engine, worker gRPC service, data loading, preprocessing, UDF execution,
  aggregation server and clients, worker landscape aggregator.
- Audited the statistics cores against scipy/statsmodels formulas and probed
  edge cases (single-observation groups, constant columns, large test
  statistics, one-sided alternatives).
- Measured instead of guessing for performance work (startup profile, per-call
  rewrite cost, `np.add.at` vs `bincount`).
- Every bug fix has a regression test that fails on `master`.
