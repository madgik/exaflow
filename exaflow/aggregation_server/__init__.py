# aggregation_server/__init__.py
import os
import tomllib
from importlib.resources import open_binary
from types import SimpleNamespace

from exaflow.grpc_options import DEFAULT_GRPC_MAX_MESSAGE_MB
from exaflow.grpc_options import grpc_max_message_mb
from exaflow.grpc_tls import load_tls_config

_DEFAULTS = {
    "port": 50051,
    "health_port": 0,
    "federation": "",
    "max_grpc_connections": 10,
    "max_wait_for_aggregation_inputs": 60,
    "log_level": "INFO",
    "max_message_mb": DEFAULT_GRPC_MAX_MESSAGE_MB,
}
_INT_KEYS = {
    "port",
    "health_port",
    "max_grpc_connections",
    "max_wait_for_aggregation_inputs",
}
_TLS_DEFAULTS = {
    "enabled": "false",
    "ca_cert_path": "",
    "cert_path": "",
    "key_path": "",
    "require_client_auth": "true",
}


def _env_value(value, default):
    """Resolve a TOML ``$ENV_VAR`` placeholder, leaving literal values alone."""
    if isinstance(value, str) and value.startswith("$"):
        return os.getenv(value[1:], default)
    return value


def _load_config():
    cfg_path = os.getenv("AGG_SERVER_CONFIG_FILE")
    if cfg_path:
        with open(cfg_path, "rb") as f:
            raw = tomllib.load(f)
    else:
        with open_binary(__package__, "config.toml") as f:
            raw = tomllib.load(f)

    cfg = {}
    for key, val in raw.items():
        if key == "tls":
            continue
        val = _env_value(val, _DEFAULTS.get(key, ""))
        if key in _INT_KEYS:
            if isinstance(val, str):
                val = val.strip()
            val = int(_DEFAULTS[key] if val in ("", None) else val)
        elif key == "max_message_mb":
            # Same rules as the controller and workers: blank means the default.
            val = grpc_max_message_mb(val)
        cfg[key] = val

    # A custom AGG_SERVER_CONFIG_FILE may predate the health/federation/TLS
    # keys; always expose the full namespace to the server.
    for key, default in _DEFAULTS.items():
        cfg.setdefault(key, default)

    tls_raw = raw.get("tls") or {}
    cfg["tls"] = load_tls_config(
        {
            key: _env_value(val, _TLS_DEFAULTS.get(key, ""))
            for key, val in tls_raw.items()
        },
        require_client_auth_default=True,
    )

    return SimpleNamespace(**cfg)


config = _load_config()
