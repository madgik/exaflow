# Scaffold Templates and Placement Rules

## Standalone Subfolder Inference

Priority order:

1. Explicit algorithm-to-subfolder rules.
1. Prefix-based fallback rules.
1. Final fallback: `tests/standalone_tests/federated_algorithms/_generated/`.

## Generated File Contracts

### Algorithm module

- Path: `exaflow/algorithms/exareme3/<algorithm>.py`
- Contains:
  - `Algorithm` subclass
  - `get_specification()` with matching `name`
  - `run()` placeholder

### Standalone test

- Path: `tests/standalone_tests/federated_algorithms/<subfolder>/test_<algorithm>.py`
- Contains a placeholder test raising `NotImplementedError`.

### Prod test

- Path: `tests/prod_env_tests/test_<algorithm>_validation.py`
- Uses existing request helpers and expected fixture path.

### Prod expected fixture

- Path: `tests/prod_env_tests/expected/<algorithm>_expected.json`
- Shape:

```json
{
  "test_cases": []
}
```

### Documentation

- Path: `documentation/algorithms/<algorithm>.md`
- Snake_case naming for new placeholders.
