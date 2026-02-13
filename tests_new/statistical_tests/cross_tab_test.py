from library.statistical_tests.cross_tab.cross_tab import CrossTab
from tests_new.utils.datasets.titanic import TitanicDataset
import numpy as np

from library.descriptive_stats.statistical_function import FederatedStatistics
from library.utils.aggregators.numpy_aggregator import NumpyAggregator
from tests_new.utils.datasets.iris import IrisDataset
from tests_new.utils.interfaces.federation_tester import FederationTester
import pandas as pd

class CrossTableTest(FederationTester):
    @staticmethod
    def federated_computation(client, local_dataset, *, column1=None, column2=None, dropna=False, **kwargs):
        aggregator = NumpyAggregator(client)
        cross_table = CrossTab(aggregator)
        output = cross_table.compute(dataset=local_dataset, column1=column1,
                                     column2=column2,dropna=dropna)
        return output


    @staticmethod
    def centralized_computation(centralized_dataset, *, column1=None, column2=None,dropna=False, **kwargs):
        cross_tab = pd.crosstab(centralized_dataset[column1],
                                centralized_dataset[column2],
                                dropna=dropna)
        return cross_tab

    @staticmethod
    def compare(federated_outputs, global_output, **kwargs):
        federated_output = federated_outputs[0]
        try:
            np.testing.assert_allclose(federated_output, global_output)
            print("✓ Outputs are equal within default tolerance!")
        except AssertionError as e:
            print("✗ Outputs differ:")
            print(e)


if __name__ == "__main__":
    dataset = TitanicDataset()
    for column1 in ['Sex', 'Embarked','Age']:
        for column2 in ['Sex', 'Embarked', 'Age']:
            for dropna in [True, False]:
                print(f'********Testing {column1}-{column2}-{dropna}********')
                CrossTableTest.run_test(
                    dataset=TitanicDataset(),
                    column1=column1,
                    column2=column2,
                    client_count=3,
                    dropna=dropna
                )

