import subprocess
import sys

from exaflow.aggregation_clients import AggregationType as ClientAggregationType
from exaflow.aggregation_types import AggregationType


def test_clients_and_server_share_one_aggregation_type_enum():
    from exaflow.aggregation_server import server

    assert ClientAggregationType is AggregationType
    assert server.AggregationType is AggregationType
    assert set(server.REDUCERS) == {
        AggregationType.SUM,
        AggregationType.MIN,
        AggregationType.MAX,
    }


def test_aggregation_type_compares_equal_to_its_wire_string():
    assert AggregationType.UNION == "UNION"
    assert AggregationType("SUM") is AggregationType.SUM
    assert str(AggregationType.MAX) == "MAX"


def test_server_imports_without_the_client_package():
    # The aggregation server image copies only aggregation_server/,
    # aggregation_types.py, aggregation_serialization.py, grpc_options.py and
    # protos/ (see exaflow/aggregation_server/Dockerfile).
    code = (
        "import sys; import exaflow.aggregation_server.server; "
        "assert 'exaflow.aggregation_clients' not in sys.modules, "
        "'server imports the client package'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
