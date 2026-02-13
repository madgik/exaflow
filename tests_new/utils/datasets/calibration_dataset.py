from tests_new.utils.interfaces.partitioned_table import PartitionedPandasTable
import pandas as pd
from pathlib import Path

CURRENT_DIR = Path(__file__).parent

class CalibrationDataset(PartitionedPandasTable):
    def get_dataset(self) -> pd.DataFrame:
        return pd.read_parquet(CURRENT_DIR / "data/calibration_dataset.parquet")

if __name__ == "__main__":
    print(CalibrationDataset().get_dataset())