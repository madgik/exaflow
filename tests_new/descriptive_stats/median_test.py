import numpy as np

from library.descriptive_stats.median import MedianBasedOnHistogram
from library.utils.aggregators.numpy_aggregator import NumpyAggregator
from tests_new.utils.interfaces.federation_tester import FederationTester


class MedianTest(FederationTester):

    @staticmethod
    def federated_computation(client, local_dataset, *, range_acc=None,column=None):
        aggregator = NumpyAggregator(client)
        local_dataset = local_dataset[column].to_numpy()
        median_value = MedianBasedOnHistogram(aggregator).compute(local_dataset, range_acc=range_acc)
        return median_value

    @staticmethod
    def centralized_computation(centralized_dataset,*, range_acc=None,column=None):
        column_values = centralized_dataset[column]
        value_range = column_values.max() - column_values.min()
        return value_range,centralized_dataset[column].median()

    @staticmethod
    def compare(federated_outputs, global_output,*, range_acc=None,**kwargs):
        value_range,global_median = global_output
        tolerance = value_range/range_acc
        for fed_median in federated_outputs:
            np.testing.assert_allclose(fed_median, global_median, rtol=tolerance, err_msg="Median Test Failed")
        print("\n[SUCCESS] Matches centralized ground truth!")


if __name__ == "__main__":
    from tests_new.utils.datasets.job_training import JobTrainingDataset
    MedianTest.run_test(dataset=JobTrainingDataset(), column='age',range_acc=0.1)