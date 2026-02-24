from library.statistical_tests.parametric.chi_squared import ChiSquared
from library.utils.aggregators.numpy_aggregator import NumpyAggregator
from tests_new.utils.datasets.titanic import TitanicDataset
from tests_new.utils.interfaces.federation_tester import FederationTester
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

class ChiSquaredTest(FederationTester):
    @staticmethod
    def federated_computation(client, local_dataset, *, factor = None, factor_categories = None, outcome = None , outcome_categories =None, **kwargs):
        aggregator = NumpyAggregator(client)
        chi_squared = ChiSquared(aggregator)
        output = chi_squared.compute(dataset=local_dataset, factor=factor, outcome=outcome,
                                     factor_categories=factor_categories, outcome_categories=outcome_categories)
        return output

    @staticmethod
    def centralized_computation(centralized_dataset, *, factor = None, factor_categories = None, outcome = None , outcome_categories =None, **kwargs):

        cross_tab = pd.crosstab(centralized_dataset[factor],
                                centralized_dataset[outcome],
                                dropna=True)

        chi2, p, dof, expected = chi2_contingency(cross_tab)
        return chi2, p, dof, expected

    @staticmethod
    def compare(federated_outputs, global_output, **kwargs):
        # All clients should return the same federated output
        federated_output = federated_outputs[0]
        chi2_fed, p_fed, dof_fed, expected_fed = federated_output
        chi2_glob, p_glob, dof_glob, expected_glob = global_output

        print(f"Federated Chi2: {chi2_fed}, P-value: {p_fed}, DOF: {dof_fed}")
        print(f"Global Chi2: {chi2_glob}, P-value: {p_glob}, DOF: {dof_glob}")

        # print(f"Federated Expected:\n{expected_fed}")
        # print(f"Global Expected:\n{expected_glob}")

        try:
            np.testing.assert_allclose(chi2_fed, chi2_glob, rtol=1e-5)
            np.testing.assert_allclose(p_fed, p_glob, rtol=1e-5)
            np.testing.assert_equal(dof_fed, dof_glob)
            np.testing.assert_allclose(expected_fed, expected_glob, rtol=1e-5)
            print("✓ Chi-Squared results are equal within tolerance!")
        except AssertionError as e:
            print("✗ Chi-Squared results differ:")
            print(e)
            raise e

if __name__ == "__main__":
    dataset = TitanicDataset()
    df = dataset.get_dataset()
    
    # Selecting some columns for testing
    # Factor and outcome must be categorical or discrete
    test_cases = [
        # ('Sex', 'Survived'),
        # ('Pclass', 'Survived'),
        ('Embarked', 'Survived')
    ]

    for factor in ['Sex','Pclass','Embarked']:
        for outcome in [ 'Survived','Embarked']:
            print(f"\n******** Testing {factor} vs {outcome} ********")

            # Pre-calculating categories as required by the current ChiSquared implementation
            ChiSquaredTest.run_test(
                dataset=TitanicDataset(),
                factor=factor,
                outcome=outcome,
                client_count=3
            )
