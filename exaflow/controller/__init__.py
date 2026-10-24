import os
from enum import Enum
from enum import unique
from importlib.resources import open_binary

import envtoml

from exaflow import AttrDict
from exaflow import controller
from exaflow.grpc_options import grpc_max_message_mb

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


# Initializing the configurations from the config file
config = None
if config_file := os.getenv("EXAFLOW_CONTROLLER_CONFIG_FILE"):
    with open(config_file, "rb") as fp:
        config = AttrDict(envtoml.load(fp))
else:
    with open_binary(controller, "config.toml") as fp:
        config = AttrDict(envtoml.load(fp))

_ensure_grpc_message_config(config)

worker_landscape_aggregator = None
