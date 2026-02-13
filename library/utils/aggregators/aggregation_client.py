from abc import ABC,abstractmethod


class AggregationClientInterface(ABC):

    @abstractmethod
    def __global_sum__(self, local_sum):
        pass

    @abstractmethod
    def __global_min__(self, local_min):
        pass

    @abstractmethod
    def __global_max__(self, local_max):
        pass

    @abstractmethod
    def __global_union__(self, categories,c_type):
        pass


