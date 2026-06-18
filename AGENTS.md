# Agent Instructions

This file is the canonical repository guidance for AI coding agents and human
reviewers. Read it before changing code, tests, documentation, deployment
manifests, or agent skills.

Durable reference material lives under `documentation/context/` — start with
`documentation/context/README.md` for the file index. Use this file for
non-negotiable rules, algorithm workflow, and scope; use the context directory
for full command lists, architecture detail, and validation matrices.

## Project Overview

Exaflow is the execution engine behind the Medical Informatics Platform. It
orchestrates federated classical algorithms across multiple workers that hold
DuckDB-backed datasets. The main runtime is a Python 3.10 backend/service system
with a Quart/Hypercorn controller, gRPC workers, optional aggregation server,
algorithm implementations, Kubernetes deployment manifests, and a nested
`exadata-validator` Python package.

Unless a task explicitly targets Flower or SMPC, focus on the native Exareme3
federated pipeline described below. Flower and SMPC are optional add-ons with
separate config, markers, and review requirements — see
`documentation/context/risk-register.md`.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `exaflow/controller/quart` | HTTP API endpoints; `endpoints.py` drives `/algorithms` and metadata routes. |
| `exaflow/controller/services` | Controller-side validation, worker landscape aggregation, execution strategies, and task handlers. |
| `exaflow/worker` | Worker gRPC service, DuckDB loading, UDF execution, and worker metadata services. |
| `exaflow/aggregation_server` | Optional gRPC aggregation service for federated vector operations. |
| `exaflow/algorithms/exareme3` | Runtime algorithm wrappers, preprocessing steps, specifications, and UDF registrations. |
| `exaflow/algorithms/federated` | Testable federated/statistical core implementations and implementation docs. |
| `exaflow/protos` | gRPC protobuf definitions and generated Python files. |
| `tests/standalone_tests` | Fast local tests for core logic, services, UDFs, aggregation, and algorithms. |
| `tests/algorithm_validation_tests` | Environment-backed algorithm validation suites. |
| `tests/prod_env_tests` | Production-like Kubernetes/kind validation tests and expected fixtures. |
| `tests/test_data` | Data-model fixtures and CSV datasets used by runtime and tests. |
| `documentation` | Human-facing API, algorithm, user-story, setup, and context docs. |
| `documentation/context` | Durable repository context for future agents and reviewers. |
| `kubernetes` | Helm chart, values, templates, and deployment docs. |
| `exadata-validator` | Separate Poetry package for validating data-model folders using DuckDB. |
| `.agents/skills` | Canonical Exaflow algorithm scaffold and validation skills. |
| `.github/workflows` | CI for lint, standalone tests, algorithm validation, prod tests, image publishing, and validator packaging. |
| `tasks.py` | Invoke tasks for config generation, local deployment, service lifecycle, data loading, and cleanup. |
| `run_algorithm` | CLI helper that builds and posts algorithm execution requests. |

## Stack

- Language/runtime: Python `>=3.10,<3.11`.
- Package manager: Poetry; root lockfile is `poetry.lock`.
- Controller: Quart and Hypercorn.
- Worker/API transport: gRPC, protobuf, grpcio health checks.
- Local persistence: DuckDB loaded from CSV datasets.
- Data/science stack: NumPy, SciPy, pandas, scikit-learn, statsmodels, pyarrow.
- Config: TOML templates read through `envtoml` or `tomli`, with selected values
  interpolated from environment variables.
- Deployment: Dockerfiles for controller, worker, aggregation server; Helm chart
  under `kubernetes`; kind is used by prod environment tests.
- Testing: pytest, pytest-xdist, pytest-subtests, pytest-cov.
- Lint/format: CI and pre-commit include Ruff, whitespace, JSON,
  debug-statement, and mdformat hooks. Agents must not run Ruff manually.
- Nested package: `exadata-validator` has its own `pyproject.toml`,
  `poetry.lock`, tests, and package build workflow.

## Essential Commands

Root package setup and local deployment:

```bash
poetry install
cp .deployment.sample.toml .deployment.toml   # first time only
poetry run inv create-configs
poetry run inv deploy
```

Focused tests while developing (preferred for agents):

```bash
poetry run pytest -q tests/standalone_tests
poetry run pytest -q tests/standalone_tests/federated_algorithms/<family>/test_<algorithm_id>.py
```

Run an algorithm through the helper CLI:

```bash
./run_algorithm -a pca -y leftamygdala lefthippocampus -d ppmi0 -m dementia:0.1
```

Full install, build, test, deployment, and Kubernetes command lists live in
`documentation/context/commands.md`. Test strategy, CI equivalents, and the
validation matrix live in `documentation/context/testing.md`.

CI runs the full standalone suite with coverage (reference only — agents should
use the focused `-q` commands above unless explicitly asked to reproduce CI):

```bash
poetry run pytest -s -m "not smpc" --cov=exaflow --cov-report=xml:non_smpc_cov.xml tests/standalone_tests --verbosity=4
```

## Lint / Format / Typecheck

CI and commit hooks run formatting and import-order checks. Agents must not run
Ruff manually; it is noisy for this workflow and should be left to the user's
commit path or CI. If a commit hook modifies Markdown during an agent-created
commit, stage the hook changes and retry the commit.

Static typecheck command: Unknown / TODO: verify. No mypy, pyright, or pyre
configuration was detected.

## Token / Output Discipline

Agents must avoid high-output commands unless explicitly requested or necessary
to verify the change. Prefer scoped commands and quiet flags.

Do not run on your own:

- Ruff, formatters, or import sorters.
- Full verbose test suites such as `pytest -s`, `pytest -vv`, coverage runs, or
  prod environment tests.
- Broad log/template dumps such as `docker logs`, `kubectl logs`,
  `kubectl describe`, or unscoped `helm template`.
- Broad repo dumps such as `find .`, `tree`, `ls -R`, unscoped `rg --files`, or
  `cat` on large files.
- Unscoped `git diff`, `git show`, or `git log -p`.
- Dependency audit/outdated commands unless the task is about dependencies.

Use focused file reads, scoped searches, `--stat`, `-q`, and path-specific test
commands instead.

## Architecture Rules

See `documentation/context/architecture.md` and
`documentation/context/module-index.md` for full flow diagrams and module
ownership. Non-negotiable boundaries:

- HTTP routes in `exaflow/controller/quart`; DTOs/validation in
  `exaflow/controller/services/api`.
- Controller orchestration in `exaflow/controller/services` (strategies, task
  handlers).
- Worker data loading and UDF execution in `exaflow/worker`.
- Runtime wrappers and specs in `exaflow/algorithms/exareme3`; testable federated
  core in `exaflow/algorithms/federated`.
- Proto, deployment, and `exadata-validator` package boundaries unchanged without
  documented compatibility impact.

## Coding Conventions

- Match existing Python style: 88-character line length, double quotes,
  and single-line imports. Do not run Ruff manually to enforce this.
- Prefer Pydantic models for API and algorithm request/response data already
  modeled by the repo.
- New code should not be backwards compatible by default. Do not add legacy
  request shapes, aliases, fallback paths, deprecated output fields, compatibility
  shims, or silent normalization for old callers unless the user explicitly asks
  for backwards compatibility.
- Keep algorithm ids lower snake_case and use the same id for module names,
  request names, spec names, tests, fixtures, and docs unless existing display
  docs intentionally differ.
- Keep specification metadata UI-ready: compact standard `label`, one concise
  method-level `desc`, and detailed parent `documentation` for formulas,
  defaults, ranges, options, assumptions, outputs, and reference anchors.
- Keep human-facing algorithm Markdown under `documentation/algorithms/`
  technical and implementation-grounded. Use one page per runtime algorithm
  when variants differ materially; do not keep duplicate overview pages that
  repeat split algorithm pages.
- Prefer specification-level validation in `exaflow/algorithms/specifications.py`
  over ad-hoc algorithm checks when the constraint is part of the public
  algorithm contract.
- Preserve config loading through the existing config modules instead of reading
  environment variables ad hoc in business logic.
- Log operational details without printing secrets, raw sensitive datasets, or
  private environment values.
- Tests should be focused first: use standalone tests for core behavior, then
  environment-backed suites only when the changed behavior needs them.

## Forbidden Patterns

- Do not silently swallow exceptions or convert them to vague errors.
- Do not introduce global mutable runtime state unless it matches an existing
  registry/config pattern and is covered by tests.
- Do not read environment variables outside the existing config/settings layer
  unless the current module already owns config loading.
- Do not bypass API DTO validation, worker landscape aggregation, algorithm
  specifications, or execution strategies.
- Do not change public API routes, algorithm request/response shapes, protobufs,
  or Helm values without documenting compatibility impact.
- Do not add production dependencies without explaining why and updating
  lockfiles intentionally.
- Do not hand-edit generated protobuf files without also documenting/regenerating
  the source `.proto` flow.
- Do not leave scaffold placeholders, `NotImplementedError`, `__REPLACE_ME_*__`,
  or unresolved TODOs in completed algorithm work.
- Do not lower privacy thresholds, disable security boundaries, or bypass
  aggregation cleanup as a convenience.
- Do not run destructive cleanup commands against user data or Kubernetes
  resources without explicit confirmation.

## Security and Privacy Rules

- Never commit secrets, tokens, credentials, private keys, cookies, or local
  private environment values.
- Never print secrets or sensitive dataset contents in logs.
- Treat privacy thresholds, SMPC/DP config, data deletion, migrations,
  external registry credentials, and deployment credentials as human-review
  areas.
- Preserve existing security and privacy boundaries, including worker privacy
  minimum row counts and local-data protection settings.
- Treat `.deployment.toml`, config files under `configs/`, and CI secret usage
  as environment-specific. Do not broaden what is logged or committed.

## PR Expectations

Every agent change should report:

- Summary of the change.
- Files changed.
- Tests and validation commands run.
- Risk assessment, including whether deployment, privacy, API, or data contracts
  are affected. Backwards-incompatible API, request, response, or algorithm
  contract changes are acceptable by default when they keep the new behavior
  simpler and correct; document the break instead of adding compatibility paths.
- Rollback notes when the change affects runtime behavior, deployment, data, or
  public interfaces.

Keep diffs small, preserve existing architecture, and update
`documentation/context/` when conventions, commands, risks, or architecture
change.

## Definition of Done

- The diff is minimal and directly tied to the request.
- Relevant source, tests, configs, and docs were read before editing.
- Public API, algorithm, proto, config, and deployment compatibility impacts are
  documented when touched; do not preserve backwards compatibility unless the
  user explicitly requests it.
- Focused tests passed, or unrun checks are named with the reason and residual
  risk.
- No unrelated user changes were overwritten.
- No speculative features, dependencies, generated artifacts, or formatting churn
  were added.

______________________________________________________________________

## New Algorithm Quickstart

Use `documentation/new-algorithm-setup.md` as the human-facing guide. For agent
work, the canonical path is:

1. Pick a lower snake_case `<algorithm_id>` and a federated `<family>`. Common
   families include `statistics`, `linear_model`, `decomposition`,
   `naive_bayes`, `mixed_effects`, `cluster`, `model_selection`, and
   `preprocessing`. See `tests/standalone_tests/federated_algorithms/` for the
   full set of existing families.

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

1. Use `--strict` only when the prod environment is available and prod
   validation is required.

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

The pipeline below is what you usually need to reference or debug.

1. **HTTP Request Intake**

   - `run_algorithm` posts to `POST /algorithms/<algorithm_name>`.
   - Quart wiring lives in `exaflow/controller/quart/endpoints.py`. It parses
     JSON into `AlgorithmRequestDTO`, validates it against enabled specs, and
     instantiates a strategy via `get_algorithm_execution_strategy`.

1. **Strategy Selection and Metadata Prep**

   - Exaflow algorithms use `ExaflowStrategy` /
     `ExaflowWithAggregationServerStrategy`
     (`exaflow/controller/services/exareme3/strategies.py`).
   - Strategy pulls metadata through the Worker Landscape Aggregator (datasets,
     variables, CDEs) to validate that inputs exist on every worker.
   - If preprocessing requests longitudinal transforms, it calls
     `prepare_longitudinal_transformation` before algorithm execution.

1. **Algorithm Instantiation**

   - Algorithm classes are discovered by importing modules from
     `EXAREME3_ALGORITHM_FOLDERS`, defaulting to
     `./exaflow/algorithms/exareme3`, and collecting `Algorithm` /
     `PreprocessingStep` subclasses from `exaflow/__init__.py`.
   - Exareme3 specifications are defined in code:
     - Algorithms implement `@classmethod get_specification()`.
     - Preprocessing steps implement `@classmethod get_specification()`.
     - `exaflow.exareme3_algorithm_classes` and
       `exaflow.exareme3_preprocessing_step_classes` are keyed by `spec.name`.
   - Module importing is designed to be idempotent even when
     `EXAREME3_ALGORITHM_FOLDERS` points at non-package folders, avoiding
     duplicate UDF registration.

1. **Flow Engine and Worker Calls**

   - `ExaflowAlgorithmFlowEngineInterface`
     (`exaflow/controller/services/exareme3/algorithm_flow_engine_interface.py`)
     wraps parallel UDF dispatch. It retrieves worker UDF keys through
     `exareme3_registry`, injects preprocessing/raw-input metadata, and runs
     requests concurrently via `ExaflowTasksHandler.run_udf` →
     `WorkerTasksHandler` (gRPC client).
   - Each UDF is tagged with `@exareme3_udf`
     (`exaflow/algorithms/exareme3/utils/registry.py`), which registers it and
     optionally flags aggregation server support.

1. **Worker Execution Path**

   - Worker gRPC server implementation is in `exaflow/worker/grpc_server.py`.
   - Startup loads DuckDB datasets through
     `duck_db_csv_loader.load_all_csvs_from_data_folder`.
   - Worker-side `run_udf` loads selected data, applies preprocessing steps,
     enforces minimum-row privacy checks, and executes registered UDFs (usually
     through helpers in `exaflow/worker/exareme3/udf/` and
     `exaflow/algorithms/exareme3/library/`).
   - For preprocessing-aware execution, worker-side `run_udf` asks each
     preprocessing step for `required_input_variables()`. For longitudinal
     preprocessing this pulls `dataset`, `subjectid`, and `visitid` through the
     `LongitudinalPreprocessingStep` contract instead of hardcoded worker logic.
   - Results are JSON-serialisable dicts that the controller stitches into the
     algorithm-specific Pydantic model returned by `algorithm.run`.

1. **Aggregation Server**

   - UDFs that need vector aggregation set aggregation metadata in the registry.
   - `ExaflowWithAggregationServerStrategy` configures
     `ControllerAggregationClient`; workers use UDF aggregation clients to push
     partial vectors and retrieve combined results.
   - Service config sits at `exaflow/aggregation_server/config.toml`; start with
     `poetry run inv start-aggregation-server`.
   - The strategy must cleanup aggregation context after execution.

1. **Response**

   - Algorithm `.run()` returns a Pydantic model; the strategy serialises it to
     JSON and returns it as the HTTP response body.

______________________________________________________________________

## Working on Algorithms

- **Specs (Exareme3):** Algorithm and preprocessing step metadata shipped to
  clients is defined in code via `get_specification()` returning
  `AlgorithmSpecification` / `PreprocessingStepSpecification`
  (`exaflow/algorithms/specifications.py`). Update the specification method and
  implementation together.
- **Specification labels:** Keep `label` compact, standard, user-facing, and not
  implementation-specific. Prefer labels such as "Logistic Regression",
  "Two-way ANOVA", "Descriptive Statistics", and "K-means"; avoid labels such as
  "Federated Logistic Regression", "Two-way ANOVA (OLS)", or implementation
  helper names.
- **UI-facing descriptions:** Keep algorithm/preprocessing step and parameter
  `desc` values compact and focused on observable method behavior. Algorithm
  and preprocessing `desc` values should be one concise sentence suitable for
  cards/tooltips. Put detailed explanations, formulas, defaults, ranges, option
  lists, assumptions, outputs, and result interpretation in the parent
  `documentation` field.
- **Specification documentation:** Parent `documentation` should describe what
  the method computes, important input interpretation, parameter behavior and
  defaults, output contents, and a careful reference anchor where possible. Use
  "aligned with" or "methodology consistent with" for packages or methods such
  as `statsmodels`, `scipy.stats`, `scikit-learn`, or standard methodology; do
  not claim exact equivalence when defaults or implementation details differ.
- **Algorithm Markdown documentation:** Files under `documentation/algorithms/`
  should use a consistent technical structure: overview, inputs, statistical
  model/method, computation without row-level data, aggregated quantities,
  `### Federated flow` text pseudocode, technical decisions, outputs,
  validation against a reference package or method, and limitations. Inspect the
  Exareme3 wrapper, federated core, and tests before writing; output fields must
  match the actual Pydantic response model returned by `.run()`. For variants
  such as one-way vs two-way ANOVA or Gaussian vs categorical Naive Bayes, prefer
  separate pages over combined duplicate pages.
- **User-facing terminology:** Do not mention platform or execution-engine terms
  in user-facing spec text, including "Exaflow", "exareme", "MIP", "worker",
  "engine", "federated implementation", or "aggregation-server-backed". Prefer
  neutral phrasing such as "computed from aggregated sufficient statistics
  without sharing raw data" when privacy-preserving computation is relevant.
- **Input labels:** Replace raw variable labels such as `x`, `y`, `var`, or
  `vars` with UI labels such as "Outcome", "Covariates", "Variables",
  "Grouping variable", "Features", or "Additional variables".
- **Parameter descriptions:** Parameter `desc` values should explain what the
  setting does, not restate options, defaults, min/max bounds, requiredness, or
  schema shape. Prefer text like "Clipping strategy for each variable." over
  "Required dictionary mapping variables to strategies." Do not add
  `documentation` to parameter specifications; parameter details belong in the
  parent step or algorithm `documentation`. Omit explicit optional `None`
  specification arguments such as `parameters=None`, `validation=None`,
  `enumslen=None`, `default=None`, `enums=None`, `dict_values_enums=None`,
  `min=None`, and `max=None`.
- **Parameter specifications:** Prefer typed specification validation over
  ad-hoc algorithm checks. For dictionary parameters, use `dict_keys_enums` for
  allowed keys, `dict_values_enums` for allowed categorical values, and
  `dict_values_type` for typed dictionary values such as numeric folds.
- **Lint/import order:** Do not spend agent time manually adjusting import order,
  formatting-only lint, or other purely mechanical style issues. Automated tools
  handle those changes.
- **Implementations:** `exaflow/algorithms/exareme3/*.py` typically defines:
  - A class derived from `Algorithm` (or `PreprocessingStep`) exposing `run` (or
    preprocessing step helpers).
  - `@classmethod get_specification()` returning the typed spec object (prefer
    `from exaflow.algorithms import specifications as specs`).
  - Optional `@classmethod required_input_variables()` for preprocessing steps to
    declare extra columns needed at worker load time.
  - UDF helpers decorated with `@exareme3_udf`.
- **Data helpers:** `metrics.py` and `library/` hold reusable computations;
  also reuse or extend helpers under `exaflow/algorithms/utils` and
  family-specific modules before inlining SQL or math.
- **Controller integration:** Ensure runtime modules live in
  `EXAREME3_ALGORITHM_FOLDERS` so discovery can populate algorithm and
  preprocessing registries.
- **New algorithm setup:** Prefer the scaffold and validation skills over manual
  file creation. They patch common exports, create canonical fixtures/docs, and
  reject incomplete placeholders in new-algorithm mode.

______________________________________________________________________

## Testing and Validation

See `documentation/context/testing.md` for the full validation matrix. Common
entrypoints:

- `poetry run pytest -q tests/standalone_tests/federated_algorithms/<family>/test_<algorithm_id>.py`
- `poetry run pytest -q tests/prod_env_tests/test_<algorithm_id>_validation.py`

Algorithm integration gates:

- `poetry run python .agents/skills/exaflow-algorithm-scaffold/scripts/integrate_new_algorithm.py --repo-root . --algorithm <algorithm_id> --family <family> --skip-scaffold`
- `poetry run python .agents/skills/exaflow-algorithm-validate/scripts/validate_algorithms.py --repo-root . --new-algorithm <algorithm_id>`
- Add `--strict` only when prod-env runtime checks should run.

Pytest markers include `slow`, `very_slow`, `smpc`, and `smpc_cluster`. Focus
on `slow` and `very_slow` when algorithm changes might affect distributed runs.

______________________________________________________________________

## Repo Skills

Use repo-path invocation for shared algorithm skills.

Scaffold and integration gate:

```bash
poetry run python .agents/skills/exaflow-algorithm-scaffold/scripts/integrate_new_algorithm.py --repo-root . --algorithm <algorithm_id> --family <family>
poetry run python .agents/skills/exaflow-algorithm-scaffold/scripts/integrate_new_algorithm.py --repo-root . --algorithm <algorithm_id> --family <family> --skip-scaffold
```

Scaffold only:

```bash
poetry run python .agents/skills/exaflow-algorithm-scaffold/scripts/scaffold_algorithms.py --repo-root . --dry-run
poetry run python .agents/skills/exaflow-algorithm-scaffold/scripts/scaffold_algorithms.py --repo-root . --algorithms <algorithm_id> --family <family>
```

Validate required steps:

```bash
poetry run python .agents/skills/exaflow-algorithm-validate/scripts/validate_algorithms.py --repo-root .
poetry run python .agents/skills/exaflow-algorithm-validate/scripts/validate_algorithms.py --repo-root . --new-algorithm <algorithm_id>
poetry run python .agents/skills/exaflow-algorithm-validate/scripts/validate_algorithms.py --repo-root . --new-algorithm <algorithm_id> --strict
```

These skills are versioned in-repo under `.agents/skills/` and are the canonical
workflow for Exareme3 algorithm scaffolding and validation.
