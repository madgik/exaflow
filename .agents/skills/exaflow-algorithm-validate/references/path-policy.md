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

When runtime contracts differ materially, use one application page per runtime
algorithm. Each page documents request inputs/defaults, application response
fields, and preprocessing/replay integration. Outputs match the wrapper Pydantic
response model. Link to the canonical method pages under
`exaflow/algorithms/federated/docs/` for mathematical and numerical details;
the [library guide](../../../../exaflow/algorithms/federated/AGENTS.md#method-documentation)
owns their structure.

When no documentation exception exists, the canonical path is:

```text
documentation/algorithms/<algorithm>.md
```

## Legacy Suite Policy

`tests/algorithm_validation_tests` remains non-blocking by design. The validator
checks canonical standalone and prod environment tests instead.
