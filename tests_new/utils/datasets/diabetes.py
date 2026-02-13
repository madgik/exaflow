import pandas as pd
from tests_new.utils.interfaces.partitioned_table import PartitionedPandasTable
from pathlib import Path

CURRENT_DIR = Path(__file__).parent

class DiabetesDiseaseDataset(PartitionedPandasTable):
    def get_dataset(self) -> pd.DataFrame:
        return pd.read_parquet(CURRENT_DIR / "data/diabetes.parquet")

    # ⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠
    # ⚠ ATTENTION DO NOT DELETE (FUTURE REFERENCE) ⚠
    # ⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠
    # def get_dataset(self) -> pd.DataFrame:
    #     from sklearn.datasets import load_diabetes
    #     diabetes = load_diabetes()
    #     x = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    #     y = pd.Series(diabetes.target, name='target')
    #     return pd.concat([x, y], axis=1)

if __name__ == "__main__":
    print(DiabetesDiseaseDataset().get_dataset())