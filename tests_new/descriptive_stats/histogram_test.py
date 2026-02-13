import numpy as np
from library.descriptive_stats.histogram import FedHistogramSimple
from library.utils.aggregators.numpy_aggregator import NumpyAggregator
from tests_new.utils.datasets.iris import IrisDataset
from tests_new.utils.interfaces.federation_tester import FederationTester


class HistogramTest(FederationTester):
    @staticmethod
    def federated_computation(client, local_dataset, *, column=None, num_bins=None, **kwargs):
        aggregator = NumpyAggregator(client)
        hist = FedHistogramSimple(aggregator)
        x = local_dataset[column].to_numpy()
        hist, bin_edges=hist.numerical_histogram(x=x, num_bins=num_bins)
        return hist, bin_edges

    @staticmethod
    def centralized_computation(centralized_dataset, *, column=None, num_bins=None, **kwargs):
        x:np.ndarray = centralized_dataset[column].to_numpy()
        print(x.min(axis=0))
        hist, bin_edges = np.histogram(x, bins=num_bins)
        return hist, bin_edges

    @staticmethod
    def compare(federated_outputs, global_output, **kwargs):
        print("Lala")


if __name__ == "__main__":
    HistogramTest.run_test(
        dataset=IrisDataset(),
        column='sepal length (cm)',
        num_bins = 5,
        client_count=1
    )
