# Agent Instructions: Kubernetes

Follow the root `AGENTS.md` first. This directory contains the Exaflow Helm
chart, values, templates, and deployment documentation.

## Commands

Render the chart before reporting deployment-template changes as complete:

```bash
helm template kubernetes/
```

Development kind deployment is documented in `DevDeployment.md`.

Prod-like validation uses:

```bash
uv run pytest tests/prod_env_tests --verbosity=4
```

Only run prod environment tests when Docker, kind, Helm, and required local
resources are available.

## Rules

- Treat image names/tags, service ports, config mounts, volumes, node selectors,
  secrets, and worker/controller/aggregation-server wiring as high review.
- Keep Helm values and templates consistent.
- Do not hardcode local paths or credentials.
- Do not change deployment behavior without updating relevant docs under
  `kubernetes/` or `documentation/context/`.
- Human review is required for production-affecting deployment changes.
