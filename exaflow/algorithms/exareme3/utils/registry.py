from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from typing import Callable
from typing import Dict

from exaflow.utils import Singleton

"""
Lightweight registry for exareme3 UDFs.

- UDFs are registered with a stable key derived from `func.__qualname__` and
  the defining module (see `get_udf_registry_key`).
- Set `with_aggregation_server=True` when a UDF needs an aggregation server
  session injected via `agg_client` (see logistic_regression for an example).
"""


def _hash(text: str) -> str:
    return base64.b32encode(hashlib.sha1(text.encode()).digest())[:4].decode().lower()


def get_udf_registry_key(func: Callable) -> str:
    module_part = func.__module__.split(".")[-1]
    qual = func.__qualname__.replace(".", "__")
    return f"{qual}_{_hash(module_part)}"


@dataclass(frozen=True)
class UDFInfo:
    func: "Callable"
    with_aggregation_server: "bool"
    enable_lazy_aggregation: "bool"
    agg_client_name: "str"


class Exareme3Registry(metaclass=Singleton):
    def __init__(self) -> None:
        self._registry: Dict[str, UDFInfo] = {}

    def register(
        self,
        func: Callable,
        *,
        with_aggregation_server: bool = False,
        enable_lazy_aggregation: bool = False,
        agg_client_name: str = "agg_client",
    ) -> str:
        """
        Register a UDF and return its registry key. Raises if the same key is
        reused for a different function to avoid ambiguity.
        """
        key = get_udf_registry_key(func)
        if key in self._registry and self._registry[key].func is not func:
            raise ValueError(f"Duplicate registration for key {key!r}")
        self._registry[key] = UDFInfo(
            func,
            with_aggregation_server,
            enable_lazy_aggregation,
            agg_client_name,
        )
        return key

    def aggregation_server_required(self, key: str) -> bool:
        info = self._registry.get(key)
        return bool(info and info.with_aggregation_server)

    def lazy_aggregation_enabled(self, key: str) -> bool:
        info = self._registry.get(key)
        return bool(info and info.enable_lazy_aggregation)

    def agg_client_name(self, key: str) -> str:
        info = self._registry.get(key)
        return info.agg_client_name if info else "agg_client"

    def get_func(self, key: str) -> Callable:
        info = self._registry.get(key)
        return info.func

    def __repr__(self) -> str:
        return f"<Exareme3Registry {len(self._registry)} entries>"


exareme3_registry = Exareme3Registry()


def exareme3_udf(
    _func: Callable | None = None,
    *,
    with_aggregation_server: bool = False,
    enable_lazy_aggregation: bool | None = None,
    agg_client_name: str = "agg_client",
):
    """
    Decorator to register a UDF and (optionally) enable lazy aggregation.

    - with_aggregation_server: whether the UDF expects an agg_client.
    - enable_lazy_aggregation: override to force on/off lazy batching. Defaults to
      matching with_aggregation_server.
    - agg_client_name: name of the aggregation client parameter used inside the UDF.
    """

    def decorator(func: Callable) -> Callable:
        lazy_on = (
            with_aggregation_server
            if enable_lazy_aggregation is None
            else enable_lazy_aggregation
        )
        exareme3_registry.register(
            func,
            with_aggregation_server=with_aggregation_server,
            enable_lazy_aggregation=lazy_on,
            agg_client_name=agg_client_name,
        )
        return func

    if _func is None:
        return decorator

    return decorator(_func)
