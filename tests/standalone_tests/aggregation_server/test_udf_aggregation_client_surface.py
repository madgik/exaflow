"""Algorithm code may only call sum/min/max/union on `agg_client`."""

import ast
import inspect
from pathlib import Path

import pytest

import exaflow.algorithms
from exaflow.aggregation_clients.exareme3_udf_aggregation_client import (
    Exareme3UDFAggregationClient,
)
from exaflow.algorithms.federated.utils.agg_client import AggregationClient
from exaflow.worker.exareme3.lazy_aggregation.lazy_aggregation import (
    LazyAggregationClient,
)

ALGORITHM_API = {"sum", "min", "max", "union"}


def _public_methods(cls) -> set[str]:
    return {
        name
        for name, _ in inspect.getmembers(cls, inspect.isfunction)
        if not name.startswith("_")
    }


def test_protocol_is_exactly_the_algorithm_api():
    assert _public_methods(AggregationClient) == ALGORITHM_API


def test_udf_client_exposes_only_the_algorithm_api_and_worker_lifecycle():
    # `unregister` and `close` are called by the worker (udf_service), not by algorithms.
    assert _public_methods(Exareme3UDFAggregationClient) == ALGORITHM_API | {
        "unregister",
        "close",
    }


@pytest.mark.parametrize("name", ["aggregate", "aggregate_batch", "unregister"])
def test_lazy_client_does_not_forward_other_client_methods(name):
    class _Client:
        def _aggregate_batch(self, ops):
            return [values for _, values in ops]

        def unregister(self):
            raise AssertionError("must not be reachable through the lazy client")

    lazy = LazyAggregationClient(_Client())

    assert lazy.sum([1, 2]) == [1, 2]
    with pytest.raises(AttributeError):
        getattr(lazy, name)


def _is_agg_client(node) -> bool:
    # `agg_client` in UDFs, `self.agg_client` / `self._agg_client` in federated cores.
    if isinstance(node, ast.Name):
        return node.id == "agg_client"
    return isinstance(node, ast.Attribute) and node.attr in {
        "agg_client",
        "_agg_client",
    }


def _agg_client_misuses(source: str) -> list[tuple[int, str]]:
    return [
        (node.lineno, node.attr)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Attribute)
        and _is_agg_client(node.value)
        and node.attr not in ALGORITHM_API
    ]


def test_misuse_scan_catches_bare_and_attribute_agg_client():
    source = (
        "agg_client.aggregate(x)\n"
        "self.agg_client.unregister()\n"
        "self._agg_client._aggregate_batch(ops)\n"
        "self.agg_client.sum(x)\n"
        "self._agg_client = agg_client\n"
    )
    assert _agg_client_misuses(source) == [
        (1, "aggregate"),
        (2, "unregister"),
        (3, "_aggregate_batch"),
    ]


def test_algorithm_code_only_calls_the_algorithm_api_on_agg_client():
    root = Path(exaflow.algorithms.__file__).parent
    offenders = [
        f"{path.relative_to(root)}:{lineno} .{attr}"
        for path in root.rglob("*.py")
        for lineno, attr in _agg_client_misuses(path.read_text())
    ]
    assert offenders == []
