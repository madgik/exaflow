import pandas as pd
from scipy.stats import chi2_contingency

from exaflow.algorithms.federated.statistics.cross_tabular_table import CrossTabTable
from exaflow.algorithms.federated.utils.aggregators.numpy_aggregator import (
    NumpyAggregator,
)


class ChiSquaredTestResult(tuple):
    def __new__(cls, chi2, p, dof, expected, x_labels, y_labels):
        return super().__new__(cls, (chi2, p, dof, expected))

    def __init__(self, chi2, p, dof, expected, x_labels, y_labels):
        self.x_labels = x_labels
        self.y_labels = y_labels


class ChiSquared:
    """
    Computes the Chi-Squared test of independence for federated datasets.
    """

    def __init__(self, aggregator: NumpyAggregator):
        """
        Initializes the ChiSquared test with a federated numpy aggregator.

        Args:
            aggregator (NumpyAggregator): The aggregator used to combine local results.
        """
        self.aggregator = aggregator

    def compute(
        self,
        dataset: pd.DataFrame,
        factor: str,
        outcome: str,
        *,
        factor_categories,
        outcome_categories,
        dropna=False,
    ):
        """
        Computes the federated Chi-Squared test.

        Args:
            dataset (pd.DataFrame): The local dataset containing the factor and outcome columns.
            factor (str): The name of the column representing the independent variable.
            outcome (str): The name of the column representing the dependent variable.
            factor_categories (list): The complete list of expected categories for the factor.
            outcome_categories (list): The complete list of expected categories for the outcome.
            dropna (bool, optional): If False, includes NaN as a separate category. Defaults to False.

        Returns:
            tuple: A tuple containing:
                - chi2 (float): The test statistic.
                - p (float): The p-value of the test.
                - dof (int): Degrees of freedom.
                - expected (ndarray): The expected frequencies, based on the marginal sums of the table.
        """
        cross_tab_table = CrossTabTable(self.aggregator).compute(
            dataset,
            factor=factor,
            outcome=outcome,
            factor_categories=factor_categories,
            outcome_categories=outcome_categories,
            dropna=dropna,
        )
        chi2, p, dof, expected = chi2_contingency(cross_tab_table)
        x_labels = [str(x) for x in cross_tab_table.index.tolist()]
        y_labels = [str(y) for y in cross_tab_table.columns.tolist()]
        return ChiSquaredTestResult(chi2, p, dof, expected, x_labels, y_labels)
