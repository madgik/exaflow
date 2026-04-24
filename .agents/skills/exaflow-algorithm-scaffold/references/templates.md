# Scaffold Templates and Placement Rules

## Standalone Subfolder Inference

Priority order:

1. Explicit `--subfolder`.
1. `--family` inferred subfolder.
1. Algorithm-name inference rules.
1. Final fallback: `tests/standalone_tests/federated_algorithms/_generated/` (reported as warning).

## Generated File Contracts

### Exareme3 algorithm module

- Path: `exaflow/algorithms/exareme3/<algorithm>.py`
- Contains:
  - `Algorithm` subclass
  - `get_specification()` with matching `name`
  - family-aware inputdata defaults
  - local UDF placeholder (or federated-core call when scaffolded)

### Federated core module (enabled by default)

- Path: `exaflow/algorithms/federated/<family>/<algorithm>.py`
- Contains:
  - `Federated<Algorithm>` placeholder class
  - `compute()` placeholder to be replaced with real logic

### Standalone test

- Path: `tests/standalone_tests/federated_algorithms/<subfolder>/test_<algorithm>.py`
- Contains placeholder test to replace with parity checks.

### Prod test

- Path: `tests/prod_env_tests/test_<algorithm>_validation.py`
- Uses existing request helpers and expected fixture path.

### Prod expected fixture

- Path: `tests/prod_env_tests/expected/<algorithm>_expected.json`
- Default shape is non-empty sample fixture:

```json
{
  "test_cases": [
    {
      "input": {
        "inputdata": {
          "y": ["__REPLACE_ME_Y__"],
          "x": ["__REPLACE_ME_X__"],
          "data_model": "__REPLACE_ME_DATA_MODEL__",
          "datasets": ["__REPLACE_ME_DATASET__"],
          "filters": null
        },
        "parameters": {}
      },
      "output": {}
    }
  ]
}
```

### Documentation

- Canonical algorithm doc: `documentation/algorithms/<algorithm>.md`
- Federated docs index target: `exaflow/algorithms/federated/docs/<algorithm>.md`

### Optional auto-patching

When enabled, scaffold updates:

- `exaflow/algorithms/federated/<family>/__init__.py`
- `exaflow/algorithms/federated/__init__.py`
- `exaflow/algorithms/specifications.py` (`AlgorithmName`)
- `exaflow/algorithms/federated/README.md`
