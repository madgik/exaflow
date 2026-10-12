import numpy as np

from exaflow.worker.exareme3.lazy_aggregation import LazyAggregationRewriter
from exaflow.worker.exareme3.lazy_aggregation import RecordingAggClient
from exaflow.worker.exareme3.lazy_aggregation import lazy_agg


@lazy_agg()
def outer_with_inner(agg_client):
    @lazy_agg()
    def inner(client):
        x = client.sum(np.array([1.0], dtype=float))
        y = client.sum(np.array([2.0], dtype=float))
        return float(np.asarray(x, dtype=float)[0] + np.asarray(y, dtype=float)[0])

    a = agg_client.sum(np.array([3.0], dtype=float))
    b = inner(agg_client)
    return float(np.asarray(a, dtype=float)[0]), b


def test_nested_functions_are_rewritten_and_batched():
    agg = RecordingAggClient()
    a, b = outer_with_inner(agg)

    assert a == 3.0
    assert b == 3.0
    # Inner batches two sums and outer single sum; order may reorder but both must appear.
    assert agg.calls == [("batch", 2), ("batch", 1)]


def test_rewrite_does_not_bind_names_in_function_globals(tmp_path):
    source = (
        "import numpy as np\n"
        "\n"
        "def udf(agg_client):\n"
        "    a = agg_client.sum(np.array([1.0]))\n"
        "    b = agg_client.sum(np.array([2.0]))\n"
        "    return float(a[0]), float(b[0])\n"
    )
    module_path = tmp_path / "udf_module.py"
    module_path.write_text(source)
    namespace = {"__name__": "udf_module"}
    exec(compile(source, str(module_path), "exec"), namespace)
    udf = namespace.pop("udf")

    rewritten = LazyAggregationRewriter().rewrite(udf)
    agg = RecordingAggClient()

    assert rewritten(agg) == (1.0, 2.0)
    assert agg.calls == [("batch", 2)]
    # The shared namespace must not gain (or lose) bindings during the rewrite.
    assert "udf" not in namespace
