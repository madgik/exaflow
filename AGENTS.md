# Codex Init — Exaflow Algorithms

Read this first whenever you open the repository through `/init`. It distills
how Exaflow’s native federated pipeline (ignoring Flower/SMPC add-ons) is
wired: where algorithms live, which configs they need, how requests travel from
the CLI to workers, and which commands are essential.

______________________________________________________________________

## Orientation

- **Purpose:** Exaflow is the execution engine behind the Medical Informatics
  Platform. It orchestrates *classical* algorithms across multiple
  workers that hold DuckDB-fed datasets.
- **Primary stack:** Python 3.10, Poetry, Quart + Hypercorn REST controller,
  gRPC worker services, DuckDB for local storage, optional aggregation server for
  vector aggregations.
- **Key dirs to inspect:**
  | Path | Why it matters |
  | --- | --- |
  | `exaflow/controller/quart` | HTTP endpoints; `endpoints.py` drives `/algorithms`. |
  | `exaflow/controller/services/exareme3` | Controller-side strategy + worker task abstractions. |
  | `exaflow/algorithms/exareme3` | Algorithm + preprocessing step implementations and in-code specs (`get_specification`). |
  | `exaflow/algorithms/federated` | Federated core modules, family exports, and implementation docs. |
  | `exaflow/worker` | gRPC server, DuckDB loader, UDF runner. |
  | `exaflow/aggregation_server` | Optional microservice providing SUM/MIN/MAX aggregation. |
  | `tasks.py` | `invoke` tasks for configs, data seeding, service lifecycle. |
  | `tests/standalone_tests/federated_algorithms` | Fast parity tests for federated core logic. |
  | `tests/prod_env_tests` | Prod validation request tests and expected fixtures. |
  | `.agents/skills` | Canonical scaffold and validation automation for algorithm work. |

______________________________________________________________________

## New Algorithm Quickstart

Use `documentation/new-algorithm-setup.md` as the human-facing guide. For agent
work, the canonical path is:

1. Pick a lower snake_case `<algorithm_id>` and a federated `<family>` such as
   `statistics`, `linear_model`, `decomposition`, or `naive_bayes`.

1. Run the integration driver once to scaffold and expose the first failing
   gates:

   ```bash
   poetry run python .agents/skills/exaflow-algorithm-scaffold/scripts/integrate_new_algorithm.py --repo-root . --algorithm <algorithm_id> --family <family>
   ```

1. Implement the generated files:

   - `exaflow/algorithms/exareme3/<algorithm_id>.py`
   - `exaflow/algorithms/federated/<family>/<algorithm_id>.py`
   - `tests/standalone_tests/federated_algorithms/<family>/test_<algorithm_id>.py`
   - `tests/prod_env_tests/test_<algorithm_id>_validation.py`
   - `tests/prod_env_tests/expected/<algorithm_id>_expected.json`
   - `documentation/algorithms/<algorithm_id>.md`
   - `exaflow/algorithms/federated/docs/<algorithm_id>.md`

1. Re-run the integration driver after edits and keep fixing until it reports
   `"done": true`:

   ```bash
   poetry run python .agents/skills/exaflow-algorithm-scaffold/scripts/integrate_new_algorithm.py --repo-root . --algorithm <algorithm_id> --family <family> --skip-scaffold
   ```

1. Use `--strict` only when the prod environment is available and prod validation
   is required.

Do not leave scaffold placeholders in new algorithms. `TODO`,
`NotImplementedError`, `__REPLACE_ME_*__` fixture values, missing docs, missing
exports, and validator warnings are incomplete work for new-algorithm mode.

Prompt template for future new-algorithm requests:

```text
Use the exaflow-algorithm-scaffold and exaflow-algorithm-validate skills to
integrate <algorithm_id> as a new federated <family> algorithm in this Exaflow
repo. Do not stop at placeholders. Implement the Exareme3 wrapper, federated
core, registrations, standalone parity tests, prod validation fixture/test, and
algorithm docs. Start with integrate_new_algorithm.py, then rerun it with
--skip-scaffold after edits until it reports "done": true. Use --strict only if
the prod environment is available.
```

______________________________________________________________________

## Deep Dive: Exaflow Algorithm Lifecycle

The pipeline below is what you usually need to reference/debug.

1. **HTTP Request Intake**

   - `run_algorithm` (root script) posts to `POST /algorithms/<algorithm_name>`.
   - Quart wiring lives in `exaflow/controller/quart/endpoints.py`. It parses JSON
     into `AlgorithmRequestDTO`, validates it against enabled specs, and instantiates a
     strategy via `get_algorithm_execution_strategy`.

1. **Strategy Selection & Metadata Prep**

   - Exaflow algorithms use `ExaflowStrategy` / `ExaflowWithAggregationServerStrategy`
     (`exaflow/controller/services/exareme3/strategies.py`).
   - Strategy pulls metadata through the Worker Landscape Aggregator (datasets,
     variables, CDES) to validate that inputs exist on every worker.
   - If preprocessing requests longitudinal transforms, it calls
     `prepare_longitudinal_transformation` before algorithm execution.

1. **Algorithm Instantiation**

   - Algorithm classes are discovered by importing modules from
     `EXAREME3_ALGORITHM_FOLDERS` (defaults to `./exaflow/algorithms/exareme3`) and
     collecting `Algorithm` / `PreprocessingStep` subclasses (`exaflow/__init__.py`).
   - Exareme3 specifications are defined in code:
     - Algorithms implement `@classmethod get_specification() -> AlgorithmSpecification`.
     - Preprocessing Steps implement `@classmethod get_specification() -> PreprocessingStepSpecification`.
     - `exaflow.exareme3_algorithm_classes` and `exaflow.exareme3_preprocessing_step_classes` are keyed by `spec.name`.
   - Module importing is designed to be idempotent even when `EXAREME3_ALGORITHM_FOLDERS`
     points at non-package folders (to avoid double-executing modules and duplicate UDF registrations).
   - Strategy creates the algorithm class, passing:
     - `Inputdata` payload (datasets, vars, parameters)
     - `ExaflowAlgorithmFlowEngineInterface` (see below)
     - Controller-side parameters

1. **Flow Engine & Worker Calls**

   - `ExaflowAlgorithmFlowEngineInterface` (`exaflow/controller/services/exareme3/algorithm_flow_engine_interface.py`)
     wraps parallel UDF dispatch. It:
     - Retrieves the correct worker UDF key through the `exareme3_registry`.
     - Injects preprocessing/raw-input metadata to each UDF call.
     - Runs requests concurrently via a thread pool, calling
       `ExaflowTasksHandler.run_udf` which in turn proxies to
       `WorkerTasksHandler` (gRPC client).
   - Each UDF is tagged with `@exareme3_udf` (see `exaflow/algorithms/exareme3/exareme3_registry.py`)
     which registers it and (optionally) flags whether aggregation server support is required.

1. **Worker Execution Path**

   - Worker gRPC server implementation is at `exaflow/worker/grpc_server.py`.
   - After startup it eagerly loads DuckDB datasets via
     `duck_db_csv_loader.load_all_csvs_from_data_folder`.
   - `WorkerTasksHandler` calls `RunUdf` on the worker; `udf_service` loads the
     registered UDF, applies parameters, and runs queries locally (usually through
     helpers in `exaflow/worker/exareme3/udf/` and `exaflow/algorithms/exareme3/library/`).
   - For preprocessing-aware execution, worker-side `run_udf` now asks each
     preprocessing step for `required_input_variables()`. For longitudinal
     preprocessing this pulls `dataset`, `subjectid`, and `visitid` through the
     `LongitudinalPreprocessingStep` contract instead of hardcoded worker logic.
   - Results are JSON-serialisable dicts that the controller stitches into the
     algorithm-specific Pydantic model returned by `algorithm.run`.

1. **Aggregation Server (optional but part of Exaflow)**

   - Some UDFs set `with_aggregation_server=True`. `ExaflowWithAggregationServerStrategy`
     wraps execution with `ControllerAggregationClient` so the workers can push partial
     vectors to the gRPC aggregation service and retrieve the combined result.
   - Service config sits at `exaflow/aggregation_server/config.toml`; start it with
     `inv start-aggregation-server`.

1. **Response**

   - Algorithm `.run(metadata)` returns a Pydantic model; the strategy serialises it
     to JSON and returns it as the HTTP response body.

______________________________________________________________________

## Working on Algorithms

- **Specs (Exareme3):** Algorithm + preprocessing step metadata shipped to clients is defined in code via
  `get_specification()` returning `AlgorithmSpecification` / `PreprocessingStepSpecification`
  (`exaflow/algorithms/specifications.py`). Update the specification method and implementation together.
- **Implementations:** `exaflow/algorithms/exareme3/*.py` typically define:
  - A class derived from `Algorithm` (or `PreprocessingStep` for preprocessing) exposing `run` (or preprocessing step helpers).
  - `@classmethod get_specification()` returning the typed spec object (prefer `from exaflow.algorithms import specifications as specs`).
  - Optional `@classmethod required_input_variables()` for preprocessing steps to
    declare extra columns needed at worker load time.
  - UDF helpers decorated with `@exareme3_udf`.
- **Data helpers:** `metrics.py` and `library/` hold reusable computations;
  prefer extending them before inlining SQL.
- **Controller integration:** Ensure the algorithm module lives in
  `EXAREME3_ALGORITHM_FOLDERS` (default `./exaflow/algorithms/exareme3`) so
  `exareme3_algorithm_classes` / `exareme3_preprocessing_step_classes` can discover it.
- **New algorithm setup:** Prefer the scaffold and validation skills over manual
  file creation. They patch common exports, create canonical fixtures/docs, and
  reject incomplete placeholders in new-algorithm mode.

______________________________________________________________________

## Testing & Validation

- **Pytest entrypoints:**
  - `poetry run pytest -q tests/standalone_tests/federated_algorithms/<family>/test_<algorithm_id>.py` — fast federated-core parity checks.
  - `poetry run pytest -q tests/prod_env_tests/test_<algorithm_id>_validation.py` — prod validation checks when the environment is available.
- **Algorithm integration gates:**
  - `poetry run python .agents/skills/exaflow-algorithm-scaffold/scripts/integrate_new_algorithm.py --repo-root . --algorithm <algorithm_id> --family <family> --skip-scaffold`
  - `poetry run python .agents/skills/exaflow-algorithm-validate/scripts/validate_algorithms.py --repo-root . --new-algorithm <algorithm_id>`
  - Add `--strict` only when prod-env runtime checks should run.
- **Markers:** Defined in `pyproject.toml` (`slow`, `very_slow`, `smpc`, etc.).
  Focus on `slow/very_slow` when algorithm changes might affect distributed runs.

______________________________________________________________________

## Repo Skills (Algorithm Workflow)

Use repo-path invocation for the shared algorithm skills:

- Scaffold and integration gate:
  - `poetry run python .agents/skills/exaflow-algorithm-scaffold/scripts/integrate_new_algorithm.py --repo-root . --algorithm <algorithm_id> --family <family>`
  - `poetry run python .agents/skills/exaflow-algorithm-scaffold/scripts/integrate_new_algorithm.py --repo-root . --algorithm <algorithm_id> --family <family> --skip-scaffold`
- Scaffold only:
  - `poetry run python .agents/skills/exaflow-algorithm-scaffold/scripts/scaffold_algorithms.py --repo-root . --dry-run`
  - `poetry run python .agents/skills/exaflow-algorithm-scaffold/scripts/scaffold_algorithms.py --repo-root . --algorithms <algorithm_id> --family <family>`
- Validate required steps:
  - `poetry run python .agents/skills/exaflow-algorithm-validate/scripts/validate_algorithms.py --repo-root .`
  - `poetry run python .agents/skills/exaflow-algorithm-validate/scripts/validate_algorithms.py --repo-root . --new-algorithm <algorithm_id>`
  - `poetry run python .agents/skills/exaflow-algorithm-validate/scripts/validate_algorithms.py --repo-root . --new-algorithm <algorithm_id> --strict`

These skills are versioned in-repo under `.agents/skills/` and are intended as the canonical workflow for Exareme3 algorithm scaffolding and validation.
