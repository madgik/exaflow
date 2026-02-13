from tests_new.utils.interfaces.partitioned_table import PartitionedPandasTable
import pandas as pd
from pathlib import Path

CURRENT_DIR = Path(__file__).parent

class IrisDataset(PartitionedPandasTable):

    def get_dataset(self) -> pd.DataFrame:
        return pd.read_parquet(CURRENT_DIR / "data/iris.parquet")

    # ⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠
    # ⚠ ATTENTION DO NOT DELETE (FUTURE REFERENCE) ⚠
    # ⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠
    # def get_dataset(self) -> pd.DataFrame:
    #     from sklearn.datasets import load_iris
    #     import numpy as np
    #     iris = load_iris()
    #     rng = np.random.RandomState(42)  # For reproducibility
    #     shuffled_indices = rng.permutation(len(iris.data))
    #
    #     iris.data = iris.data[shuffled_indices]
    #     iris.target = iris.target[shuffled_indices]
    #
    #     df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    #     df['target'] = iris.target
    #
    #     # Filter to only use two classes (0 and 1)
    #     df = df[df['target'] != 2]
    #     df.to_parquet(CURRENT_DIR / "data/iris.parquet")
    #     return df

if __name__ == "__main__":
    print(IrisDataset().get_dataset())
