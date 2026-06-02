# Decision Log

Use this file for durable architectural decisions. Mark inferred decisions as
`Inferred / verify` until a maintainer confirms them.

**Do not treat `Inferred / verify` entries as enforced policy.** Prefer
`AGENTS.md`, source code, and `documentation/api-specification.md` when they
conflict with an inferred entry here.

## Template

### Decision: <title>

Status: Proposed / Accepted / Deprecated / Inferred / verify

Context:

Decision:

Consequences:

Files affected:

Date:

## Inferred Decisions

### Decision: Keep Federated Core Logic Separately Testable

Status: Inferred / verify

Context: Runtime wrappers live under `exaflow/algorithms/exareme3`, while core
federated implementations and standalone parity tests live under
`exaflow/algorithms/federated`.

Decision: Algorithm math should remain in federated core modules where possible,
with Exareme3 wrappers handling runtime orchestration and specifications.

Consequences: New algorithm work usually needs both a runtime wrapper and a
standalone-tested federated core.

Files affected: `exaflow/algorithms/exareme3`,
`exaflow/algorithms/federated`, `tests/standalone_tests/federated_algorithms`.

Date: 2026-05-22

### Decision: Use Runtime Specifications As Algorithm Form Source Of Truth

Status: Inferred / verify

Context: `documentation/api-specification.md` states that `GET /algorithms` is
the runtime source of truth for algorithm availability and form shape.

Decision: Clients and agents should not hardcode algorithm request fields that
are already represented by algorithm specifications.

Consequences: Specification changes must be tested and documented as public
contract changes.

Files affected: `exaflow/algorithms/specifications.py`,
`exaflow/controller/services/api`, `documentation/api-specification.md`.

Date: 2026-05-22

### Decision: Use Repository Skills For New Algorithm Work

Status: Inferred / verify

Context: Root `AGENTS.md` and `documentation/new-algorithm-setup.md` identify
repository skills as the canonical new-algorithm workflow.

Decision: Agents should use scaffold and validation skill scripts rather than
manually creating every integration file.

Consequences: New algorithm work is not complete until skill validation passes
without failures or warnings.

Files affected: `.agents/skills`, `documentation/new-algorithm-setup.md`,
algorithm wrapper/core/test/docs paths.

Date: 2026-05-22
