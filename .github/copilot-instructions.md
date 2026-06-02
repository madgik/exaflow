# GitHub Copilot Repository Instructions

Follow the repository rules in `AGENTS.md`.

Before making changes, read `documentation/context/README.md` for the file
index, then open the context files that match your task:

- `documentation/context/architecture.md` — runtime flow and module boundaries.
- `documentation/context/module-index.md` — directory ownership and key files.
- `documentation/context/commands.md` — install, run, build, test, deployment.
- `documentation/context/testing.md` — test strategy and validation matrix.
- `documentation/context/conventions.md` — coding patterns and repository norms.
- `documentation/context/risk-register.md` — sensitive or high-review areas.
- `documentation/context/code-review-checklist.md` — review checklist.
- `documentation/context/ai-task-templates.md` — reusable task prompts.
- `documentation/context/decision-log.md` — architectural decisions (see status
  field; do not treat `Inferred / verify` entries as settled policy).

For every change:

- Keep diffs small.
- Preserve existing architecture.
- Add or update tests when behavior changes.
- Run the relevant validation commands.
- Explain risks and verification steps.
