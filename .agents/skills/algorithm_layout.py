"""Runtime-domain placement shared by algorithm integration tools.

This module deliberately does not import Exaflow: scaffold dry runs must also
work before runtime dependencies or a new algorithm are available.
"""

import re
from pathlib import Path

RUNTIME_DOMAINS = {"outlier_report": "outliers"}
FEDERATED_FAMILIES = {"outlier_report": "outliers"}


def runtime_module_path(
    repo_root: Path,
    algorithm: str,
    *,
    family: str | None = None,
    domain: str | None = None,
) -> Path:
    """Find the sole existing wrapper, or choose a domain for a new one."""
    base = repo_root / "exaflow" / "algorithms" / "exareme3"
    filename = algorithm + ".py"
    matches = sorted(base.rglob(filename))
    if len(matches) > 1:
        paths = ", ".join(str(path.relative_to(repo_root)) for path in matches)
        raise ValueError(f"Ambiguous runtime modules for '{algorithm}': {paths}")
    if matches:
        path = matches[0]
        if domain and path.parent != base / domain:
            raise ValueError(
                f"Runtime domain '{domain}' conflicts with existing module {path}."
            )
        return path
    selected_domain = domain or RUNTIME_DOMAINS.get(algorithm) or family
    if not selected_domain or not re.fullmatch(r"[a-z][a-z0-9_]*", selected_domain):
        raise ValueError("Provide --runtime-domain or --family for a new wrapper.")
    return base / selected_domain / filename


def runtime_import_path(module_path: Path, repo_root: Path) -> str:
    return ".".join(module_path.relative_to(repo_root).with_suffix("").parts)
