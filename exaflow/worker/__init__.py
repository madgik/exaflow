import os
from importlib.resources import open_binary
from pathlib import Path

import envtoml

from exaflow import AttrDict
from exaflow import worker


def _ensure_duckdb_config(cfg: AttrDict) -> None:
    duckdb_cfg = getattr(cfg, "duckdb", AttrDict({}))
    duckdb_path = duckdb_cfg.get("path")
    if not duckdb_path:
        default_path = Path(cfg.data_path) / f"data_models.duckdb"
        duckdb_path = str(default_path)
    duckdb_cfg.path = duckdb_path
    cfg.duckdb = duckdb_cfg


def _ensure_grpc_config(cfg: AttrDict) -> None:
    grpc_cfg = getattr(cfg, "grpc", AttrDict({}))

    def _normalize_port(value, fallback):
        candidate = value
        if isinstance(candidate, str):
            candidate = candidate.strip()
        if candidate in ("", None):
            candidate = fallback
        try:
            return int(candidate)
        except (TypeError, ValueError):
            return int(fallback)

    default_ip = getattr(getattr(cfg, "grpc", AttrDict({})), "ip", "0.0.0.0")
    default_port_raw = getattr(getattr(cfg, "grpc", AttrDict({})), "port", 50051)
    default_port = _normalize_port(default_port_raw, 50051)

    ip_value = grpc_cfg.get("ip", default_ip)
    grpc_cfg.ip = ip_value or default_ip

    port_value = grpc_cfg.get("port", default_port)
    grpc_cfg.port = _normalize_port(port_value, default_port)

    cfg.grpc = grpc_cfg


def _ensure_worker_tasks_config(cfg: AttrDict) -> None:
    worker_tasks_cfg = getattr(cfg, "worker_tasks", AttrDict({}))

    tasks_timeout = worker_tasks_cfg.get("tasks_timeout", 120)
    if isinstance(tasks_timeout, str):
        tasks_timeout = tasks_timeout.strip()
    if tasks_timeout in ("", None):
        tasks_timeout = 120
    try:
        worker_tasks_cfg.tasks_timeout = int(tasks_timeout)
    except (TypeError, ValueError):
        worker_tasks_cfg.tasks_timeout = 120

    cfg.worker_tasks = worker_tasks_cfg


def _ensure_log_levels(cfg: AttrDict) -> None:
    def _sanitize(value, default="INFO"):
        if isinstance(value, str):
            value = value.strip()
        return value or default

    cfg.log_level = _sanitize(getattr(cfg, "log_level", "INFO"))
    cfg.framework_log_level = _sanitize(getattr(cfg, "framework_log_level", "INFO"))


if config_file := os.getenv("EXAFLOW_WORKER_CONFIG_FILE"):
    with open(config_file, "rb") as fp:
        config = AttrDict(envtoml.load(fp))
else:
    with open_binary(worker, "config.toml") as fp:
        config = AttrDict(envtoml.load(fp))

_ensure_duckdb_config(config)
_ensure_grpc_config(config)
_ensure_worker_tasks_config(config)
_ensure_log_levels(config)
