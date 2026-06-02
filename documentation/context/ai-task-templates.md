# AI Task Templates

## Bug Investigation Template

Goal:
Investigate and propose or implement a minimal fix for the bug below.

Before editing, read:

- `AGENTS.md`
- `documentation/context/architecture.md`
- `documentation/context/module-index.md`
- `documentation/context/testing.md`
- Relevant source files and tests

Bug:
...

Constraints:

- Keep the fix minimal.
- Preserve public API behavior unless explicitly requested.
- Add a regression test when possible.

Return:

1. Root cause
1. Evidence
1. Fix
1. Files changed
1. Tests run
1. Risk assessment

## Refactor Plan Template

Goal:
Plan a behavior-preserving refactor for the area below.

Before editing, read:

- `AGENTS.md`
- `documentation/context/architecture.md`
- `documentation/context/module-index.md`
- Existing tests for the area

Area:
...

Constraints:

- Do not change public API, algorithm specs, protobufs, or deployment behavior
  unless explicitly requested.
- Prefer small reviewable steps.

Return:

1. Current structure
1. Proposed target structure
1. Step-by-step change plan
1. Tests to run after each step
1. Risks and rollback notes

## Test Generation Template

Goal:
Add focused tests for the behavior below.

Before editing, read:

- `documentation/context/testing.md`
- Nearby test modules
- Source code under test

Behavior:
...

Constraints:

- Prefer standalone tests unless runtime wiring is the behavior under test.
- Keep fixtures minimal.
- Do not update expected JSON unless behavior intentionally changed.

Return:

1. Tests added
1. Cases covered
1. Commands run
1. Remaining gaps

## Code Review Template

Goal:
Review the current diff for bugs, regressions, test gaps, and architecture
violations.

Before reviewing, read:

- `AGENTS.md`
- `documentation/context/risk-register.md`
- `documentation/context/code-review-checklist.md`

Focus:

- Correctness
- Security/privacy
- Public contracts
- Tests
- Deployment risk

Return:

1. Findings ordered by severity with file/line references
1. Open questions
1. Test gaps
1. Brief change summary

## Dependency Update Template

Goal:
Assess or perform the dependency update below.

Before editing, read:

- Root or nested `pyproject.toml`
- Relevant lockfile
- `.github/workflows/*`
- `documentation/context/risk-register.md`

Dependency:
...

Constraints:

- Do not update unrelated dependencies.
- Update lockfiles intentionally.
- For scientific stack updates, expect possible numerical fixture changes and
  justify them.

Return:

1. Dependency impact
1. Files changed
1. Tests run
1. Compatibility risks
1. Rollback notes

## Backend/API Feature Template

Goal:
Implement the backend/API feature below.

Before editing, read:

- `documentation/api-specification.md`
- `documentation/context/architecture.md`
- `exaflow/controller/quart/endpoints.py`
- Relevant DTOs/validators under `exaflow/controller/services/api`

Feature:
...

Constraints:

- Keep request validation explicit.
- Document public contract changes.
- Add focused controller/API tests.

Return:

1. Behavior implemented
1. Public interface changes
1. Tests run
1. Risks and rollback notes

## Algorithm Work Template

Goal:
Add or modify the Exareme3 algorithm below.

Before editing, read:

- `AGENTS.md`
- `documentation/new-algorithm-setup.md`
- Relevant `.agents/skills/*/SKILL.md`
- Nearby algorithm wrappers, federated core modules, tests, and docs

Algorithm:
...

Constraints:

- Use repository scaffold/validation skills for new algorithms.
- Keep algorithm id consistent everywhere.
- Prefer specification-level validation.
- Do not leave placeholders.

Return:

1. Algorithm contract
1. Files changed
1. Validation skill output
1. Tests run
1. Remaining risks

## Security-Sensitive Change Template

Goal:
Plan or implement the security/privacy-sensitive change below.

Before editing, read:

- `documentation/context/risk-register.md`
- Relevant config, deployment, privacy, SMPC/DP, or logging code

Change:
...

Constraints:

- Do not weaken existing boundaries without explicit approval.
- Do not print or expose secrets.
- Include rollback notes.

Return:

1. Security impact
1. Human-review areas
1. Implementation summary
1. Tests/validation run
1. Rollback plan

## Documentation Update Template

Goal:
Update repository documentation for the change below.

Before editing, read:

- `AGENTS.md`
- `documentation/context/README.md`
- Existing docs in the affected area

Change:
...

Constraints:

- Keep docs evidence-based.
- Mark unknowns as `Unknown / TODO: verify`.
- Avoid duplicating large sections across instruction files.

Return:

1. Files changed
1. New or changed guidance
1. Unknowns/TODOs
1. Validation run

## Architecture Review Template

Goal:
Review the architecture of the area below and identify improvement options.

Before editing, read:

- `documentation/context/architecture.md`
- `documentation/context/module-index.md`
- `documentation/context/risk-register.md`
- Relevant source and tests

Area:
...

Constraints:

- Do not make product-code changes unless explicitly requested.
- Separate confirmed facts from recommendations.

Return:

1. Current architecture
1. Strengths
1. Risks
1. Recommended changes
1. Validation strategy
