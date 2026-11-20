# Validator Path Policy

The validator enforces deterministic paths for algorithm implementation,
tests, expected fixtures, and documentation. New algorithms should use the
canonical paths reported by the validator.

Documentation is the only category with explicit filename exceptions. Existing
algorithm documentation uses display-style filenames such as `FisherExact.md`,
`LinearRegression.md`, and `k-means.md`, and split technical pages such as
`ANOVAOneWay.md`, `ANOVATwoWay.md`, and `NaiveBayesGaussian.md`, so the
validator keeps those paths in `PREFERRED_DOC_PATHS` in
`scripts/validate_algorithms.py`.

When runtime variants differ materially, prefer one technical page per runtime
algorithm rather than a combined duplicate overview page. Each page should
document inputs, method/model, computation without row-level data, aggregated
quantities, federated-flow pseudocode, technical decisions, outputs, validation
reference, and limitations. Outputs must match the wrapper Pydantic response
model.

When no documentation exception exists, the canonical path is:

```text
documentation/algorithms/<algorithm>.md
```

## Preprocessing coverage

Preprocessing steps use their own registry and specification, not the analysis
algorithm enum. Standalone tests can live under
`tests/standalone_tests/exareme3/<domain>/test_<step>.py` or in a grouped test
that explicitly imports the step module or names its ID. Production coverage
can likewise be grouped under `tests/prod_env_tests`; all referencing test files
are executed in strict mode. Each step still needs its own public documentation
under `documentation/algorithms/<step>.md`.

Analysis algorithms retain the canonical test and expected-fixture requirements
above. Executable preprocessing assertions do not need a fabricated analysis
result fixture. Unrelated tests and comments do not count as step references.

## Legacy Suite Policy

`tests/algorithm_validation_tests` remains non-blocking by design. The validator
checks canonical standalone and prod environment tests instead.
