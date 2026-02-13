import pandas as pd
import numpy as np

from tests_new.utils.interfaces.partitioned_table import PartitionedPandasTable


class CoxDataset(PartitionedPandasTable):
    def get_dataset(self) -> pd.DataFrame:

        e_array = np.array([1,1,0,1,1,1,1])
        times_array = np.array([4,3,3,4,2,0,2])
        var_1 = [-1,-2,-3,-4,-5,-6,-7]
        var_2 = [-1,-2,-3,-4,-5,-6,-7]

        covariates = np.column_stack((var_1,var_2))
        times = times_array + 1
        failed = e_array
        data = {
            'e' : e_array,
            'times' : times,
            'var1' : var_1,
            'var2' : var_2
        }
        # Create DataFrame
        df = pd.DataFrame(data)
        return df
