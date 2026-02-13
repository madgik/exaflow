import pandas as pd
import numpy as np
from tests_new.utils.interfaces.partitioned_table import PartitionedPandasTable

class DummyDataset(PartitionedPandasTable):
    def get_dataset(self) -> pd.DataFrame:
        np.random.seed(42)
        data = {
            'A': np.random.randint(0, 100, size=25).tolist(),  # 0-99
            'B': np.random.randint(100, 200, size=25).tolist(),  # 100-199
            'C': np.random.randint(500, 1000, size=25).tolist(),  # 500-999
            'D': np.random.randint(-50, 50, size=25).tolist()  # -50 to 49
        }
        # Create DataFrame
        df = pd.DataFrame(data)
        return df

if __name__ == "__main__":
    print(DummyDataset().get_dataset())