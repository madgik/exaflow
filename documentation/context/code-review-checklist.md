# Code Review Checklist

## Correctness

- [ ] Does the change solve the stated problem?
- [ ] Are edge cases handled?
- [ ] Are numerical/statistical results justified when algorithm behavior
  changes?
- [ ] Are public API, algorithm spec, protobuf, config, or deployment changes
  documented?

## Tests

- [ ] Are relevant tests added or updated?
- [ ] Were the most focused validation commands run first?
- [ ] Are unrun tests named with reasons and residual risk?
- [ ] Are expected fixtures changed only for intended behavior changes?

## Architecture

- [ ] Does the change respect controller, worker, aggregation, algorithm, and
  deployment boundaries?
- [ ] Is federated core logic kept testable outside the full runtime when
  practical?
- [ ] Is configuration handled through existing config modules?
- [ ] Are generated files consistent with their sources?

## Security and Privacy

- [ ] No secrets are committed.
- [ ] Sensitive data is not logged.
- [ ] Worker privacy thresholds and local-data protections are preserved or
  explicitly reviewed.
- [ ] SMPC, DP, deletion, deployment credential, and registry changes
  received human review.

## Maintainability

- [ ] The change is small enough to review.
- [ ] Naming and structure match the repo.
- [ ] No unrelated refactors or formatting churn were included.
- [ ] Documentation/context files are updated if commands, conventions,
  architecture, or risks changed.

## Release Risk

- [ ] Rollback considerations are included for runtime/deployment changes.
- [ ] Docker/Helm changes were rendered or otherwise validated.
- [ ] Dependency updates include lockfile changes and focused tests.
- [ ] `exadata-validator` package changes were tested from its package root.
