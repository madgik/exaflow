import pandas as pd
from tests_new.utils.interfaces.partitioned_table import PartitionedPandasTable
from pathlib import Path

CURRENT_DIR = Path(__file__).parent

class HeartDisease(PartitionedPandasTable):
    def get_dataset(self) -> pd.DataFrame:
        return pd.read_parquet(CURRENT_DIR / "data/heart_disease.parquet")

    # ⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠
    # ⚠ ATTENTION DO NOT DELETE (FUTURE REFERENCE) ⚠
    # ⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠
    # def get_dataset(self) -> pd.DataFrame:
    #     import kagglehub
    #     from kagglehub import KaggleDatasetAdapter
    #     dataset = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS,"fedesoriano/heart-failure-prediction","heart.csv")
    #     dataset.to_parquet(CURRENT_DIR / "data/heart_disease.parquet")
    #     return dataset

if __name__ == "__main__":
    print(HeartDisease().get_dataset())