# Validator Path Policy

The validator enforces deterministic paths for algorithm implementation,
tests, expected fixtures, and documentation. New algorithms should use the
canonical paths reported by the validator.

Documentation is the only category with explicit filename exceptions. Existing
algorithm documentation uses display-style filenames such as `ANOVA.md`,
`FisherExact.md`, `LinearRegression.md`, and `k-means.md`, so the validator keeps
those paths in `PREFERRED_DOC_PATHS` in
`scripts/validate_algorithms.py`.

When no documentation exception exists, the canonical path is:

```text
documentation/algorithms/<algorithm>.md
```

## Legacy Suite Policy

`tests/algorithm_validation_tests` remains non-blocking by design. The validator
checks canonical standalone and prod environment tests instead.
