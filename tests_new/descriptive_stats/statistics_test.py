import numpy as np
import scipy.stats as stats

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
            "count": fed_stats.fed_count(x),
            "avg": fed_stats.mean(x), # Default ddof=0
            "var_ddof1": fed_stats.variance(x, ddof=1),
            "std_ddof1": fed_stats.standard_deviation(x, ddof=1),
            "var_ddof0": fed_stats.variance(x, ddof=0),
            "std_ddof0": fed_stats.standard_deviation(x, ddof=0),
            "ss": fed_stats.sum_of_squares(x),
            "range": fed_stats.range(x),
            "cv": fed_stats.coefficient_of_variation(x),
            "mad": fed_stats.mean_absolute_deviation(x),
            "rms": fed_stats.root_mean_square(x),
            "ms": fed_stats.mean_square(x),
            "cov_ddof1": fed_stats.covariance(x, y, ddof=1),
            "cov_ddof0": fed_stats.covariance(x, y, ddof=0),
            "pearsonr": fed_stats.pearson_correlation(x, y),
            "linregress": fed_stats.least_squares_regression(x, y),
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
            "count": float(n_x),
            "avg": mean_x,
            "var_ddof1": np.var(x, ddof=1),
            "std_ddof1": np.std(x, ddof=1),
            "var_ddof0": np.var(x, ddof=0),
            "std_ddof0": np.std(x, ddof=0),
            "ss": np.sum(x ** 2),
            "range": np.max(x) - np.min(x),
            "cv": np.std(x, ddof=0) / mean_x if mean_x != 0 else 0,
            "mad": np.mean(np.abs(x - mean_x)),
            "rms": np.mean(np.abs(x)),
            "ms": np.var(x, ddof=0),
            "cov_ddof1": np.cov(x, y, ddof=1)[0, 1],
            "cov_ddof0": np.mean((x - np.mean(x)) * (y - np.mean(y))),
            "pearsonr": stats.pearsonr(x, y)[0],
            "linregress": stats.linregress(x, y)[:2],
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
                
        print("\n[SUCCESS] ScipyStatsTest matches centralized ground truth!")

if __name__ == "__main__":
    ScipyStatsTest.run_test(
        dataset=IrisDataset(),
        client_count=3,
        column1='sepal length (cm)',
        column2='sepal width (cm)',
        operation_id=101
    )
