from library.utils.aggregators.aggregation_client import AggregationClientInterface
from abc import ABC, abstractmethod


class StatisticalModel(ABC):

    def __init__(self, client: AggregationClientInterface):
        self.client = client

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass