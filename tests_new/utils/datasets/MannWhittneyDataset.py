import numpy as np

from tests_new.utils.interfaces.partitioned_table import PartitionedPandasTable
import pandas as pd
import random


class MannWhittneyDataset(PartitionedPandasTable):
    def get_dataset(self) -> pd.DataFrame:
        df = pd.DataFrame()
        random.seed(42)
        group_a = np.asarray([random.randint(80, 140) for _ in range(100)])
        group_b = np.asarray([random.randint(80, 140) for _ in range(100)])
        df['x'] = group_a
        df['y'] = group_b
        return df

if __name__ == "__main__":
    print(MannWhittneyDataset().get_dataset())