# Compatibility Mapping (Transition)

This skill keeps legacy naming compatibility to avoid false failures while the repository converges on canonical naming conventions.

## Policy

- Standard mode:
  - Canonical path missing + legacy path present => `legacy_used` warning (non-failing).
- `--new-algorithm` mode:
  - Canonical path is mandatory.
  - Legacy path may still be detected, but canonical missing is reported as `canonical_missing` (failing).

## Prod Test Compatibility

Maps selected algorithms to legacy `tests/prod_env_tests` filenames when
`test_<algorithm>_validation.py` is not present.

## Prod Expected Compatibility

Maps selected algorithms to legacy expected fixture names when
`<algorithm>_expected.json` is not present.

## Documentation Compatibility

Maps selected algorithms to legacy documentation files such as
`ANOVA.md`, `LinearRegression.md`, or `k-means.md` when
`documentation/algorithms/<algorithm>.md` is not present.

## Standalone Compatibility

`linear_regression` currently maps to `test_ols.py`.

## Legacy Suite Policy

`tests/algorithm_validation_tests` remains non-blocking by design.
