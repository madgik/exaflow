# aggregation_server/__init__.py
import os
import tomllib
from importlib.resources import open_binary
from types import SimpleNamespace

from exaflow.grpc_options import DEFAULT_GRPC_MAX_MESSAGE_MB
from exaflow.grpc_options import grpc_max_message_mb

_DEFAULTS = {
    "port": 50051,
    "max_grpc_connections": 10,
    "max_wait_for_aggregation_inputs": 60,
    "log_level": "INFO",
    "max_message_mb": DEFAULT_GRPC_MAX_MESSAGE_MB,
}
_INT_KEYS = {
    "port",
    "max_grpc_connections",
    "max_wait_for_aggregation_inputs",
}


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
        if isinstance(val, str) and val.startswith("$"):
            env_name = val[1:]
            val = os.getenv(env_name, _DEFAULTS[key])
        if key in _INT_KEYS:
            val = int(val)
        elif key == "max_message_mb":
            # Same rules as the controller and workers: blank means the default.
            val = grpc_max_message_mb(val)
        cfg[key] = val

    return SimpleNamespace(**cfg)


config = _load_config()
