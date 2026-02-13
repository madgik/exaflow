from exaflow.aggregation_clients import AggregationType
from exaflow.aggregation_clients.exareme3_udf_aggregation_client import Exareme3UDFAggregationClient
from library.utils.aggregators.aggregation_client import AggregationClientInterface

class ExaflowAggregationClient(Exareme3UDFAggregationClient, AggregationClientInterface):
    """
    Implementation of AggregationClientInterface using Exareme3UDFAggregationClient.
    This bridges the high-level aggregation interface used in the library with 
    the Exaflow aggregation server.
    """

    def __global_sum__(self, local_sum):
        return self.aggregate(AggregationType.SUM, local_sum)

    def __global_min__(self, local_min):
        return self.aggregate(AggregationType.MIN, local_min)

    def __global_max__(self, local_max):
        return self.aggregate(AggregationType.MAX, local_max)

    def __global_union__(self, categories, c_type=None):
        # The interface includes c_type, which is not used by the current 
        # aggregation server implementation for UNION.
        return self.aggregate(AggregationType.UNION, categories)
