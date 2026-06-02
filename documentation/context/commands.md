# Commands

Command reference for Exaflow. For agent rules and algorithm workflow, start
with `AGENTS.md` at the repository root.

## Package Manager

Detected: Poetry for the root package and the nested `exadata-validator`
package.

Evidence:

- Root `pyproject.toml` and `poetry.lock`.
- `exadata-validator/pyproject.toml` and `exadata-validator/poetry.lock`.
- CI uses `snok/install-poetry` and `poetry install`.

## Install

Root package:

```bash
poetry install
```

Optional shell:

```bash
poetry shell
```

Optional hooks:

```bash
pre-commit install
```

Nested validator package:

```bash
cd exadata-validator
poetry install
```

## Run Locally

Create local deployment config:

```bash
cp .deployment.sample.toml .deployment.toml
poetry run inv create-configs
```

Deploy all local services:

```bash
poetry run inv deploy
```

Deploy without installing dependencies:

```bash
poetry run inv deploy --no-install-dep
```

Start services individually:

```bash
poetry run inv start-controller
poetry run inv start-worker --worker localworker1
poetry run inv start-aggregation-server
```

Attach to service output:

```bash
poetry run inv attach --controller
poetry run inv attach --worker localworker1
```

Run an algorithm request:

```bash
./run_algorithm -a pca -y leftamygdala lefthippocampus -d ppmi0 -m dementia:0.1
```

Dry-run an algorithm request:

```bash
./run_algorithm -a pca -y leftamygdala lefthippocampus -d ppmi0 -m dementia:0.1 -n
```

## Build

Controller Docker image:

```bash
docker build -t <USERNAME>/exaflow_controller:<IMAGETAG> -f exaflow/controller/Dockerfile .
```

Worker Docker image:

```bash
docker build -t <USERNAME>/exaflow_worker:<IMAGETAG> -f exaflow/worker/Dockerfile .
```

Aggregation server Docker image:

```bash
docker build -f exaflow/aggregation_server/Dockerfile -t exaflow/aggregation_server:latest .
```

Nested validator package:

```bash
cd exadata-validator
poetry build
```

## Test

Fast standalone suite:

```bash
poetry run pytest -q tests/standalone_tests
```

CI standalone suite:

```bash
poetry run pytest -s -m "not smpc" --cov=exaflow --cov-report=xml:non_smpc_cov.xml tests/standalone_tests --verbosity=4
```

Focused federated algorithm test:

```bash
poetry run pytest -q tests/standalone_tests/federated_algorithms/<family>/test_<algorithm_id>.py
```

Algorithm validation suite:

```bash
poetry run inv create-configs
poetry run inv deploy --no-install-dep
poetry run pytest tests/algorithm_validation_tests/exareme3 --verbosity=4
```

Prod environment suite:

```bash
poetry run pytest tests/prod_env_tests --verbosity=4
```

Nested validator tests:

```bash
cd exadata-validator
poetry run pytest -q
```

## Lint / Format

CI and pre-commit run formatting and import-order checks. Agents must not run
Ruff manually; leave Ruff to the user's commit path or CI because its output is
noisy in agent sessions.

Pre-commit is available for humans and commit hooks, but agents should run it
only when explicitly asked.

## Typecheck

Unknown / TODO: verify. No mypy, pyright, or pyre configuration was detected in
the inspected repository files.

## Token / Output Discipline

Agents should keep command output scoped and quiet. Do not run these commands
without an explicit request or a concrete verification need:

- Ruff, formatters, or import sorters.
- Full verbose test suites such as `pytest -s`, `pytest -vv`, coverage runs, or
  prod environment tests.
- Broad log/template dumps such as `docker logs`, `kubectl logs`,
  `kubectl describe`, or unscoped `helm template`.
- Broad repo dumps such as `find .`, `tree`, `ls -R`, unscoped `rg --files`, or
  `cat` on large files.
- Unscoped `git diff`, `git show`, or `git log -p`.
- Dependency audit/outdated commands unless the task is about dependencies.

Prefer focused file reads, scoped searches, `--stat`, `-q`, and path-specific
test commands.

## Database / Migrations

No schema migration command was detected. Workers load CSV data into DuckDB.

Create DuckDB data for workers through Invoke:

```bash
poetry run inv create-duckdb --worker localworker1
```

Structure test data paths:

```bash
poetry run inv structure-data
```

## Docker

Remove local containers through Invoke:

```bash
poetry run inv rm-containers
```

Start optional SMPC deployment:

```bash
poetry run inv deploy-smpc
```

Treat cleanup/removal commands as potentially destructive.

## Kubernetes / Helm

Render chart:

```bash
helm template kubernetes/
```

Install chart:

```bash
helm install exaflow kubernetes/ --debug
```

Development kind setup is documented in `kubernetes/DevDeployment.md`.

## Algorithm Skill Commands

Scaffold/integrate a new algorithm:

```bash
poetry run python .agents/skills/exaflow-algorithm-scaffold/scripts/integrate_new_algorithm.py --repo-root . --algorithm <algorithm_id> --family <family>
```

Re-run after implementation:

```bash
poetry run python .agents/skills/exaflow-algorithm-scaffold/scripts/integrate_new_algorithm.py --repo-root . --algorithm <algorithm_id> --family <family> --skip-scaffold
```

Validate algorithm artifacts:

```bash
poetry run python .agents/skills/exaflow-algorithm-validate/scripts/validate_algorithms.py --repo-root . --new-algorithm <algorithm_id>
```

## CI

- `.github/workflows/lint.yml` runs import-order lint and format checks.
- `.github/workflows/standalone_tests.yml` installs Python 3.10 and Poetry,
  then runs standalone tests excluding SMPC.
- `.github/workflows/algorithm_validation_tests.yml` deploys local services and
  runs Exareme3 algorithm validation tests.
- `.github/workflows/prod_env_tests.yml` builds images, starts kind/Helm
  deployment, and runs prod environment tests.
- `.github/workflows/exadata_validator.yml` runs validator tests, builds the
  package, and smoke-tests the installed wheel.
- Publish workflows build/push Docker images or publish `exadata-validator`.
