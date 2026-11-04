# Repository Context

This directory contains durable context for humans and AI coding agents working
on Exaflow.

## Files

- `architecture.md`: system design, runtime flow, and boundaries.
- `module-index.md`: directory/module ownership and navigation notes.
- `commands.md`: install, run, build, test, lint, format, and deployment
  commands.
- `testing.md`: testing strategy and validation matrix.
- `conventions.md`: coding patterns and repository norms.
- `risk-register.md`: fragile, security-sensitive, or high-review areas.
- `decision-log.md`: lightweight architectural decision log.
- `code-review-checklist.md`: human and AI review checklist.
- `ai-task-templates.md`: reusable prompts for common agent tasks.

## How To Use This Context

Start with `AGENTS.md` at the repository root, then open the context file that
matches the task. `AGENTS.md` holds non-negotiable rules and algorithm workflow;
this directory holds detailed reference material (commands, architecture,
testing matrix, risks). For algorithm work, also read
`documentation/new-algorithm-setup.md` and use the skills under
`.agents/skills/`.

Keep these files current when commands, architecture, risks, or conventions
change. Mark uncertain information as `Unknown / TODO: verify` instead of
guessing.
