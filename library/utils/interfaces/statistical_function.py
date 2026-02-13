from library.utils.aggregators.aggregation_client import AggregationClientInterface
from library.utils.aggregators.numpy_aggregator import NumpyAggregator
from abc import ABC, abstractmethod


class StatisticalFunction(ABC):

    def __init__(self, client: AggregationClientInterface):
        self.client = client

    @abstractmethod
    def compute(self, *args, **kwargs):
        pass


class NumpyStatisticalFunction(ABC):

    def __init__(self, aggregator: NumpyAggregator):
        self.aggregator = aggregator
