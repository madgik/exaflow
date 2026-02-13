import pandas as pd
from tests_new.utils.interfaces.partitioned_table import PartitionedPandasTable
from pathlib import Path

CURRENT_DIR = Path(__file__).parent

class TitanicDataset(PartitionedPandasTable):
    def get_dataset(self) -> pd.DataFrame:
       return pd.read_parquet(CURRENT_DIR / "data/titanic.parquet")

    # ⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠
    # ⚠ ATTENTION DO NOT DELETE (FUTURE REFERENCE) ⚠
    # ⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠
    # def get_dataset(self) -> pd.DataFrame:
    #     url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    #     df = pd.read_csv(url)
    #     df.to_parquet(CURRENT_DIR / "data/titanic.parquet")
    #     return df

if __name__ == "__main__":
    print(TitanicDataset().get_dataset())