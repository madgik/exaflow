"""Shared gRPC message limit for every Exaflow gRPC hop.

Controller-to-worker channels, the worker server, aggregation channels, and
the aggregation server all use this limit. It applies to every RPC on those
connections, including metadata, health, and Flower calls, not only RunUdf.

``GRPC_MAX_MESSAGE_MB`` (default 100) is the configuration value, in
mebibytes. Component config files interpolate that variable and Helm sets it
on the controller, workers, and aggregation server.
"""

import os

DEFAULT_GRPC_MAX_MESSAGE_MB = 100
GRPC_MAX_MESSAGE_MB_ENV = "GRPC_MAX_MESSAGE_MB"
_MIB = 1024 * 1024

# Default send and receive limit. Use grpc_message_options() for the
# configured size.
GRPC_MAX_MESSAGE_SIZE = DEFAULT_GRPC_MAX_MESSAGE_MB * _MIB


def grpc_max_message_mb(value=None) -> int:
    """Return a message limit in mebibytes.

    An explicit config ``value`` wins. Otherwise ``GRPC_MAX_MESSAGE_MB`` is
    read, and a missing or blank value defaults to 100.
    """
    if value is None:
        value = os.environ.get(GRPC_MAX_MESSAGE_MB_ENV, "")
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return DEFAULT_GRPC_MAX_MESSAGE_MB
    if value is None:
        return DEFAULT_GRPC_MAX_MESSAGE_MB
    try:
        mb = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{GRPC_MAX_MESSAGE_MB_ENV} must be a positive integer number of "
            f"mebibytes, got {value!r}"
        ) from exc
    if mb < 1:
        raise ValueError(f"{GRPC_MAX_MESSAGE_MB_ENV} must be >= 1, got {mb}")
    return mb


def grpc_max_message_size(max_message_mb=None) -> int:
    return grpc_max_message_mb(max_message_mb) * _MIB


def grpc_message_options(max_message_mb=None):
    size = grpc_max_message_size(max_message_mb)
    return [
        ("grpc.max_send_message_length", size),
        ("grpc.max_receive_message_length", size),
    ]
