import pandas as pd
from tests_new.utils.interfaces.partitioned_table import PartitionedPandasTable
from pathlib import Path

CURRENT_DIR = Path(__file__).parent

class JobTrainingDataset(PartitionedPandasTable):

    def get_dataset(self) -> pd.DataFrame:
        return pd.read_parquet(CURRENT_DIR / "data/job_training.parquet")

    # ⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠
    # ⚠ ATTENTION DO NOT DELETE (FUTURE REFERENCE) ⚠
    # ⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠
    # def get_dataset(self) -> pd.DataFrame:
    #     import statsmodels.api as sm
    #     dataset = sm.datasets.get_rdataset("lalonde", "MatchIt").data
    #     dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    #
    #     race_dummies = pd.get_dummies(dataset["race"])
    #     dataset = dataset.drop(columns=["race"])
    #     dataset = pd.concat([dataset, race_dummies], axis=1)
    #     dataset.to_parquet(CURRENT_DIR / "data/job_training.parquet")
    #     return dataset

if __name__ == "__main__":
    print(JobTrainingDataset().get_dataset())