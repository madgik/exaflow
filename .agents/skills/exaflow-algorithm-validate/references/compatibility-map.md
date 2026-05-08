# Documentation Path Mapping

The validator enforces deterministic paths for algorithm code, tests, and
expected fixtures. Documentation is the only mapped path category because the
repository intentionally uses display-style filenames such as
`ANOVA.md`, `FisherExact.md`, `LinearRegression.md`, or `k-means.md`.

When no explicit mapping exists, the fallback is
`documentation/algorithms/<algorithm>.md`.

## Legacy Suite Policy

`tests/algorithm_validation_tests` remains non-blocking by design.
