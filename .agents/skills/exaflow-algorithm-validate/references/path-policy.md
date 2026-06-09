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

## Legacy Suite Policy

`tests/algorithm_validation_tests` remains non-blocking by design. The validator
checks canonical standalone and prod environment tests instead.
