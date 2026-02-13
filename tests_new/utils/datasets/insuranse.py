import pandas as pd
from tests_new.utils.interfaces.partitioned_table import PartitionedPandasTable
from pathlib import Path

CURRENT_DIR = Path(__file__).parent

class InsuranceDataset(PartitionedPandasTable):
    def get_dataset(self) -> pd.DataFrame:
        insurance_df = pd.read_parquet(CURRENT_DIR / "data/insurance.parquet")
        insurance_processed = pd.get_dummies(insurance_df, columns=['sex', 'smoker', 'region'], drop_first=True)
        return insurance_processed

    # ⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠
    # ⚠ ATTENTION DO NOT DELETE (FUTURE REFERENCE) ⚠
    # ⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠
    # def get_dataset(self) -> pd.DataFrame:
    #     # Load the insurance dataset (replace with your file path)
    #     url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
    #     insurance_df = pd.read_csv(url)
    #     # Convert categorical variables (sex, smoker, region) into numerical using one-hot encoding
    #     insurance_processed = pd.get_dummies(insurance_df, columns=['sex', 'smoker', 'region'], drop_first=True)
    #     return insurance_processed

if __name__ == "__main__":
    print(InsuranceDataset().get_dataset())