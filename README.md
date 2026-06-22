# Exaflow [![Maintainability](https://qlty.sh/gh/madgik/projects/exaflow/maintainability.svg)](https://qlty.sh/gh/madgik/projects/exaflow) [![Code Coverage](https://qlty.sh/gh/madgik/projects/exaflow/coverage.svg)](https://qlty.sh/gh/madgik/projects/exaflow)

### For AI agents and reviewers

Start with [`AGENTS.md`](AGENTS.md) for repository rules, algorithm workflow,
and scope. Durable reference material lives under
[`documentation/context/`](documentation/context/README.md) (architecture,
commands, testing matrix, risk register, and review checklists).

### Prerequisites

1. Install [python3.10](https://www.python.org/downloads/ "python3.10")

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/ "uv")

## Setup

#### Environment Setup

1. Install dependencies

   ```
   uv sync --all-groups
   ```

1. Activate virtual environment

   ```
   source .venv/bin/activate
   ```

1. *Optional* To install tab completion for `invoke` run (replacing `bash` with your shell)

   ```
   source <(uv run inv --print-completion-script bash)
   ```

1. _Optional_ `pre-commit` is included in development dependencies. To install hooks

   ```
   pre-commit install
   ```

#### Local Deployment

1. Create a deployment configuration file `.deployment.toml` from the sample file:

   ```
   cp .deployment.sample.toml .deployment.toml
   ```

1. Create the config files that the worker services will use

   ```
   inv create-configs
   ```

1. Install dependencies, start the containers and then the services with

   ```
   inv deploy
   ```

1. Attach to some service's stdout/stderr with

   ```
   inv attach --controller
   ```

   or

   ```
   inv attach --worker <WORKER-NAME>
   ```

1. Restart a specific worker service with

   ```
   inv start-worker --localworker1
   ```

#### Execute an analysis

- Examples
  ```
  ./run_analysis -a pca -y leftamygdala lefthippocampus -d ppmi0 -m dementia:0.1
  ```
  ```
  ./run_analysis -a pearson_correlation -y leftamygdala lefthippocampus -d ppmi0 -m dementia:0.1 -p alpha 0.95
  ```

## Algorithm Development

For a new Exareme3 federated algorithm, start with
[`documentation/new-algorithm-setup.md`](documentation/new-algorithm-setup.md).
It contains the scaffold command, validation gate, required files, and a prompt
you can give Codex for end-to-end integration.

Fast path from the repository root:

```
uv run python .agents/skills/exaflow-algorithm-scaffold/scripts/integrate_new_algorithm.py --repo-root . --algorithm <algorithm_id> --family <family>
```

After implementation edits:

```
uv run python .agents/skills/exaflow-algorithm-scaffold/scripts/integrate_new_algorithm.py --repo-root . --algorithm <algorithm_id> --family <family> --skip-scaffold
```

# Acknowledgement

This project/research received funding from the European Union’s Horizon 2020 Framework Programme for Research and Innovation under the Framework Partnership Agreement No. 650003 (HBP FPA).
