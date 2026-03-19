import numpy as np
import pandas as pd

from exaflow.algorithms.federated.utils.aggregators.numpy_aggregator import (
    NumpyAggregator,
)


class CrossTabTable:
    """
    Computes a cross-tabulation table (contingency table) for federated datasets.
    """

    def __init__(self, aggregator: NumpyAggregator):
        """
        Initializes the CrossTabTable with a federated numpy aggregator.

        Args:
            aggregator (NumpyAggregator): The aggregator used to combine local results.
        """
        self.aggregator = aggregator

    def compute(
        self,
        dataset: pd.DataFrame,
        factor,
        outcome,
        *,
        factor_categories=None,
        outcome_categories=None,
        dropna=False,
    ):
        """
        Computes the federated cross-tabulation for the given factor and outcome.

        Args:
            dataset (pd.DataFrame): The local dataset containing the factor and outcome columns.
            factor (str): The name of the column representing the independent variable.
            outcome (str): The name of the column representing the dependent variable.
            factor_categories (list): The complete list of expected categories for the factor.
            outcome_categories (list): The complete list of expected categories for the outcome.
            dropna (bool, optional): If False, includes NaN as a separate category. Defaults to False.

        Returns:
            pd.DataFrame: The globally aggregated cross-tabulation table.
        """
        # Convert columns to categorical with the full set of categories without mutating dataset
        cross_tab = pd.crosstab(dataset[factor], dataset[outcome], dropna=dropna)

        if factor_categories is None:
            factor_categories = self.aggregator.fed_union(dataset[factor].to_numpy())
        if outcome_categories is None:
            outcome_categories = self.aggregator.fed_union(dataset[outcome].to_numpy())

        # Reindex with categories including NaN
        if not dropna:
            factor_categories_null = factor_categories + [np.nan]
            outcome_categories_null = outcome_categories + [np.nan]
        else:
            factor_categories_null = factor_categories
            outcome_categories_null = outcome_categories
        cross_tab = cross_tab.reindex(
            index=pd.Index(factor_categories_null, name=factor),
            columns=pd.Index(outcome_categories_null, name=outcome),
            fill_value=0,
        )

        cross_tab.iloc[:] = self.aggregator.fed_sum(cross_tab.values)

        # Drop rows and columns that are all zeros globally, matching pandas crosstab behavior
        cross_tab = cross_tab.loc[
            (cross_tab != 0).any(axis=1), (cross_tab != 0).any(axis=0)
        ]

        if not dropna:
            expected_factor_idx = pd.Index(factor_categories, name=factor)
            if cross_tab.index.isna().any():
                expected_factor_idx = expected_factor_idx.append(
                    pd.Index([np.nan], name=factor)
                )
            cross_tab.index = expected_factor_idx[
                expected_factor_idx.isin(cross_tab.index)
            ]

            expected_outcome_idx = pd.Index(outcome_categories, name=outcome)
            if cross_tab.columns.isna().any():
                expected_outcome_idx = expected_outcome_idx.append(
                    pd.Index([np.nan], name=outcome)
                )
            cross_tab.columns = expected_outcome_idx[
                expected_outcome_idx.isin(cross_tab.columns)
            ]

        return cross_tab
