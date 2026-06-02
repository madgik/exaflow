# Agent Instructions: exadata-validator

This directory is a separate Poetry package for validating data-model folders
using DuckDB. Follow the root `AGENTS.md` first, then these package-specific
rules.

## Package Overview

`exadata-validator` exposes a CLI through `validator.commands:entry` and can
also run as `python -m validator`.

## Commands

Run commands from `exadata-validator/`.

Install:

```bash
poetry install
```

Test:

```bash
poetry run pytest -q
```

Build:

```bash
poetry build
```

Smoke test an installed package as CI does:

```bash
python -m validator validate-data-model tests/data/success/data_model_longitudinal_v_1_0
```

## Rules

- Keep dependency and lockfile changes local to this package unless the root
  package also intentionally changes.
- Preserve the CLI contract documented in `README.md`.
- Add or update tests under `tests/` for validation behavior changes.
- Treat CSV/CDE validation semantics as public behavior for package users.
- Do not mix Exaflow service deployment concerns into this package.
