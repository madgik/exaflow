import numpy as np

from library.descriptive_stats.statistical_function import FederatedStatistics
from library.utils.aggregators.numpy_aggregator import NumpyAggregator
from tests_new.utils.datasets.iris import IrisDataset
from tests_new.utils.interfaces.federation_tester import FederationTester


class ScipyStatsTest(FederationTester):
    @staticmethod
    def federated_computation(client, local_dataset, *, column1=None, column2=None, **kwargs):
        aggregator = NumpyAggregator(client)
        fed_stats = FederatedStatistics(aggregator)
        
        x = local_dataset[column1].values
        y = local_dataset[column2].values
        
        results = {
            "smd": fed_stats.standardized_mean_differences(x, y)
        }
        return results

    @staticmethod
    def centralized_computation(centralized_dataset, *, column1=None, column2=None, **kwargs):
        x = centralized_dataset[column1].values
        y = centralized_dataset[column2].values
        
        # Ground truth using numpy and scipy
        n_x = len(x)
        mean_x = np.mean(x)
        
        results = {
            "smd": (np.mean(x) - np.mean(y)) / np.sqrt(((len(x) - 1) * np.var(x, ddof=1) + (len(y) - 1) * np.var(y, ddof=1)) / (len(x) + len(y) - 2))
        }
        return results

    @staticmethod
    def compare(federated_outputs, global_output, **kwargs):
        # We only need to check one client's output since they should all reach the same global state
        federated_output = federated_outputs[0]
        
        for key in global_output:
            fed_val = federated_output[key]
            glob_val = global_output[key]
            
            print(f"Comparing {key}: Fed={fed_val}, Glob={glob_val}")
            
            if isinstance(fed_val, (tuple, list, np.ndarray)):
                np.testing.assert_allclose(fed_val, glob_val, rtol=1e-5, err_msg=f"Mismatch in {key}")
            else:
                np.testing.assert_allclose(fed_val, glob_val, rtol=1e-5, err_msg=f"Mismatch in {key}")

        print("\n[SUCCESS] Matches centralized ground truth!")

if __name__ == "__main__":
    ScipyStatsTest.run_test(
        dataset=IrisDataset(),
        client_count=3,
        column1='sepal length (cm)',
        column2='sepal width (cm)',
        operation_id=101
    )
