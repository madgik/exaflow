
import numpy as np
from scipy.stats import mannwhitneyu

from library.statistical_tests.nonparametric.mann_whitney_utest import MannWhitneyUTest
from library.utils.aggregators.numpy_aggregator import NumpyAggregator
from tests_new.utils.datasets.MannWhittneyDataset import MannWhittneyDataset
from tests_new.utils.interfaces.federation_tester import FederationTester


class MannWhitneyTester(FederationTester):

    @staticmethod
    def federated_computation(client,local_dataset, **kwargs):
        aggregator = NumpyAggregator(client)
        x_values = local_dataset['x'].values
        y_values = local_dataset['y'].values
        func = MannWhitneyUTest(aggregator)
        ans = func.compute(x_values, y_values, num_bins=100)
        return ans

    @staticmethod
    def centralized_computation( centralized_dataset, **kwargs):
        # 2. Estimate propensity scores: P(Treatment | Confounders)
        x_values = centralized_dataset['x'].values
        y_values = centralized_dataset['y'].values
        return mannwhitneyu(x_values, y_values, use_continuity=True, alternative='two-sided', method='asymptotic')

    @staticmethod
    def compare(federated_outputs, global_output, **kwargs):
        # Compare each element of the output
        fed_statistic,fed_pvalue = federated_outputs[0]
        statistic,pvalue = (global_output.statistic, global_output.pvalue)

        try:
            np.testing.assert_allclose(fed_statistic, statistic, rtol=1e-5)
            np.testing.assert_allclose(fed_pvalue, pvalue, atol=1e-3, rtol=1e-1)
            print("✓ Fisher Exact results are equal within tolerance!")
        except AssertionError as e:
            print("✗ Fisher Exact results differ:")
            print(e)
            raise e


if __name__ == "__main__":
    MannWhitneyTester.run_test(dataset=MannWhittneyDataset(), x='x', y='y',client_count=1)