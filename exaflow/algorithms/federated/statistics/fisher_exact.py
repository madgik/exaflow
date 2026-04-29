import pandas as pd
from scipy.stats import fisher_exact

from exaflow.algorithms.federated.statistics.cross_tabular_table import CrossTabTable
from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.utils.aggregators.numpy_aggregator import (
    NumpyAggregator,
)


class FisherExactTestResult(tuple):
    def __new__(cls, odds_ratio, p_value, x_labels, y_labels):
        return super().__new__(cls, (odds_ratio, p_value))

    def __init__(self, odds_ratio, p_value, x_labels, y_labels):
        self.x_labels = x_labels
        self.y_labels = y_labels


class FisherExact:
    """
    Computes Fisher's Exact Test for federated datasets.
    """

    def __init__(self, aggregator: NumpyAggregator):
        """
        Initializes the FisherExact test with a federated numpy aggregator.

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
        factor_categories=None,
        outcome_categories=None,
        dropna=False,
    ):
        """
        Computes the federated Fisher's Exact Test.

        Args:
            dataset (pd.DataFrame): The local dataset containing the factor and outcome columns.
            factor (str): The name of the column representing the independent variable.
            outcome (str): The name of the column representing the dependent variable.
            factor_categories (list): The complete list of expected categories for the factor.
            outcome_categories (list): The complete list of expected categories for the outcome.
            dropna (bool, optional): If False, includes NaN as a separate category. Defaults to False.

        Returns:
            tuple: A tuple containing:
                - odds_ratio (float): The prior odds ratio.
                - p_value (float): The p-value of the test.
        """
        cross_tab_table = CrossTabTable(self.aggregator).compute(
            dataset,
            factor=factor,
            outcome=outcome,
            factor_categories=factor_categories,
            outcome_categories=outcome_categories,
            dropna=dropna,
        )

        if cross_tab_table.shape != (2, 2):
            raise BadInputError(
                "Fisher's Exact Test is only well-defined for 2x2 tables."
            )

        odds_ratio, p_value = fisher_exact(cross_tab_table)
        x_labels = [str(x) for x in cross_tab_table.index.tolist()]
        y_labels = [str(y) for y in cross_tab_table.columns.tolist()]
        return FisherExactTestResult(odds_ratio, p_value, x_labels, y_labels)
