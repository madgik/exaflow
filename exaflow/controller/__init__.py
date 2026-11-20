import os
from enum import Enum
from enum import unique
from importlib.resources import open_binary

import envtoml

from exaflow import AttrDict
from exaflow import controller
from exaflow.grpc_options import grpc_max_message_mb
from exaflow.grpc_tls import load_tls_config

BACKGROUND_LOGGER_NAME = "controller_background_service"


@unique
class DeploymentType(str, Enum):
    LOCAL = "LOCAL"
    KUBERNETES = "KUBERNETES"


def _ensure_grpc_message_config(cfg: AttrDict) -> None:
    grpc_cfg = getattr(cfg, "grpc", None)
    if not isinstance(grpc_cfg, AttrDict):
        grpc_cfg = AttrDict({})
    grpc_cfg.max_message_mb = grpc_max_message_mb(grpc_cfg.get("max_message_mb"))
    cfg.grpc = grpc_cfg


def _ensure_log_levels(cfg: AttrDict) -> None:
    for name in ("log_level", "framework_log_level"):
        value = cfg.get(name)
        if isinstance(value, str):
            value = value.strip()
        cfg[name] = "INFO" if value is None or value == "" else value


def _ensure_aggregation_tls_config(cfg: AttrDict) -> None:
    agg_cfg = getattr(cfg, "aggregation_server", None)
    if not isinstance(agg_cfg, AttrDict):
        agg_cfg = AttrDict({})
    agg_cfg.tls = load_tls_config(agg_cfg.get("tls"))
    cfg.aggregation_server = agg_cfg


# Initializing the configurations from the config file
config = None
if config_file := os.getenv("EXAFLOW_CONTROLLER_CONFIG_FILE"):
    with open(config_file, "rb") as fp:
        config = AttrDict(envtoml.load(fp))
else:
    with open_binary(controller, "config.toml") as fp:
        config = AttrDict(envtoml.load(fp))

_ensure_grpc_message_config(config)
_ensure_log_levels(config)
_ensure_aggregation_tls_config(config)

worker_landscape_aggregator = None
