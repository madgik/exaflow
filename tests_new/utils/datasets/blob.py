import pandas as pd
from tests_new.utils.interfaces.partitioned_table import PartitionedPandasTable
from sklearn.datasets import make_blobs


class BlobDataset(PartitionedPandasTable):

    def __init__(self,*,n_samples=500, centers=3):
        # Initialize the data only once
        super().__init__(n_samples=n_samples, centers=centers)

    def get_dataset(self,*,n_samples, centers) -> pd.DataFrame:
        records, _ = make_blobs(n_samples=n_samples, centers=centers, random_state=42)
        return pd.DataFrame(records, columns=["x", "y"])