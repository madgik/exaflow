import pandas as pd
from tests_new.utils.interfaces.partitioned_table import PartitionedPandasTable
import numpy as np

class MetricDataset(PartitionedPandasTable):
    def get_dataset(self) -> pd.DataFrame:
        np.random.seed(42)
        # Generate 100 binary true labels (0 or 1)
        y_true = np.random.randint(0, 2, size=100)
        # Generate 100 predicted probabilities between 0 and 1
        y_prob:np.ndarray = np.random.rand(100)
        # Predicted
        y_pred = (y_prob >= 0.5).astype(int)
        return pd.DataFrame({'y_true': y_true, 'y_prob': y_prob, 'y_pred': y_pred})

if __name__ == "__main__":
    print(MetricDataset().get_dataset())