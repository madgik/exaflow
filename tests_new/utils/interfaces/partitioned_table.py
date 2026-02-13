
from abc import ABC,abstractmethod
import pandas as pd


class PartitionedPandasTable(ABC):

    output = dict()

    def __init__(self, *args, **kwargs):
        # Initialize the data only once
        self.dataset = self.get_dataset(*args, **kwargs)

    @abstractmethod
    def get_dataset(self, *args, **kwargs) -> pd.DataFrame:
        pass

    def get_local_dataset(self,partition_id, num_partitions) -> pd.DataFrame:
        if num_partitions ==1:
            return self.dataset.copy()
        else:
            n = len(self.dataset)
            _size = n // num_partitions  # number of rows per partition (ignores remainder)
            start = partition_id * _size
            end = (
                          partition_id + 1) * _size if partition_id < num_partitions - 1 else n  # last partition may include remainder
            local_dataset = self.dataset.iloc[start:end].copy()
            return local_dataset

    def get_global_dataset(self):
        return self.dataset