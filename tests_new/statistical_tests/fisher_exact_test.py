from library.statistical_tests.nonparametric.fisher_exact import FisherExact
from library.utils.aggregators.numpy_aggregator import NumpyAggregator
from tests_new.utils.datasets.titanic import TitanicDataset
from tests_new.utils.interfaces.federation_tester import FederationTester
import pandas as pd
import numpy as np
from scipy.stats import fisher_exact

class FisherExactTest(FederationTester):
    @staticmethod
    def federated_computation(client, local_dataset, *, factor = None, factor_categories = None, outcome = None , outcome_categories =None, **kwargs):
        aggregator = NumpyAggregator(client)
        fisher = FisherExact(aggregator)
        output = fisher.compute(dataset=local_dataset, factor=factor, outcome=outcome,
                                     factor_categories=factor_categories, outcome_categories=outcome_categories)
        return output

    @staticmethod
    def centralized_computation(centralized_dataset, *, factor = None, factor_categories = None, outcome = None , outcome_categories =None, **kwargs):

        cross_tab = pd.crosstab(centralized_dataset[factor],
                                centralized_dataset[outcome],
                                dropna=True)
        return fisher_exact(cross_tab)

    @staticmethod
    def compare(federated_outputs, global_output, **kwargs):
        # All clients should return the same federated output
        federated_output = federated_outputs[0]
        odds_ratio_fed, p_value_fed = federated_output
        odds_ratio_glob, p_value_glob = global_output

        print(f"Federated Odds Ratio: {odds_ratio_fed}, P-value: {p_value_fed}")
        print(f"Global Odds Ratio: {odds_ratio_glob}, P-value: {p_value_glob}")

        try:
            np.testing.assert_allclose(odds_ratio_fed, odds_ratio_glob, rtol=1e-5)
            np.testing.assert_allclose(p_value_fed, p_value_glob, atol=1e-3, rtol=1e-1)
            print("✓ Fisher Exact results are equal within tolerance!")
        except AssertionError as e:
            print("✗ Fisher Exact results differ:")
            print(e)
            raise e

if __name__ == "__main__":
    dataset = TitanicDataset()
    df = dataset.get_dataset()
    
    # Sex and Survived are both 2x2
    test_cases = [
        ('Sex', 'Survived'),
    ]

    for factor in ['Sex', 'Pclass', 'Embarked']:
        for outcome in ['Embarked','Survived']:
            print(f"\n******** Testing {factor} vs {outcome} ********")

            FisherExactTest.run_test(
                dataset=TitanicDataset(),
                factor=factor,
                outcome=outcome,
                client_count=3
            )
