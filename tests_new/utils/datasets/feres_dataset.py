import pandas as pd
from pathlib import Path

from tests_new.utils.interfaces.partitioned_table import PartitionedPandasTable

class FeresDataset(PartitionedPandasTable):
    def get_dataset(self) -> pd.DataFrame:
        data_path = Path(__file__).parent.parent.parent / 'data' / 'feres_analysis' / 'ssrdataset_harmonized.csv'

        print(f"Looking for file at: {data_path}")
        print(f"File exists: {data_path.exists()}")
        if data_path.exists():
            df = pd.read_csv(data_path)
            print(f"Successfully loaded! Shape: {df.shape}")
        else:
            raise FileNotFoundError(f"File not found at: {data_path}")
        return df