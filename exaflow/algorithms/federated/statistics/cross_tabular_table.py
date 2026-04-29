import numpy as np
import pandas as pd

from exaflow.algorithms.federated.utils import BadInputError
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
            factor_categories (list, optional): The complete list of expected categories for the factor.
            outcome_categories (list, optional): The complete list of expected categories for the outcome.
            dropna (bool, optional): If False, includes NaN as a separate category. Defaults to False.

        Returns:
            pd.DataFrame: The globally aggregated cross-tabulation table.
        """

        # 1. Obtain consistent indices across all workers
        if factor_categories:
            factor_idx = pd.Index(factor_categories, name=factor)
        else:
            factor_idx = self.get_global_categories(
                dataset[factor].values, factor, dropna=dropna
            )

        if outcome_categories:
            outcome_idx = pd.Index(outcome_categories, name=outcome)
        else:
            outcome_idx = self.get_global_categories(
                dataset[outcome].values, outcome, dropna=dropna
            )

        # 2. Ensure NaN is present if not dropna and NaNs exist in data
        if not dropna:
            factor_idx = self._ensure_nan_index(factor_idx, dataset[factor].values)
            outcome_idx = self._ensure_nan_index(outcome_idx, dataset[outcome].values)

        # Replaces None with NaN for consistency
        factor_idx = pd.Index(
            [np.nan if pd.isna(x) else x for x in factor_idx], name=factor_idx.name
        )
        outcome_idx = pd.Index(
            [np.nan if pd.isna(x) else x for x in outcome_idx], name=outcome_idx.name
        )
        # Ensure the indexes are unique (replaces multiple NaNs with one if they occurred)
        factor_idx = factor_idx.unique()
        outcome_idx = outcome_idx.unique()

        # 3. Compute local cross-tabulation
        # We use dropna=dropna here, but reindexing at step 3 ensures we capture NaNs if needed
        cross_tab = pd.crosstab(dataset[factor], dataset[outcome], dropna=dropna)

        # 3. Reindex local result to the unified global universe.
        # This aligns all workers' results for correct matrix summation.
        cross_tab = cross_tab.reindex(
            index=factor_idx, columns=outcome_idx, fill_value=0
        )
        global_values = self.aggregator.fed_sum(cross_tab.values)
        cross_tab.iloc[:, :] = global_values

        # 4. Prune rows and columns that are entirely zero.
        # This is essential when categories are forced from metadata but not present in the data slice.
        cross_tab = cross_tab.loc[
            (cross_tab != 0).any(axis=1), (cross_tab != 0).any(axis=0)
        ]

        if cross_tab.empty or cross_tab.size == 0:
            raise BadInputError("Contingency table is empty after pruning.")

        return cross_tab

    # We need a robust way to obtain global unions that is consistent across all workers.
    # NumpyAggregator.fed_union can be brittle for distributed NaNs due to local decisions.
    # We manually synchronize categories and NaN presence here to ensure matrix alignment.
    def get_global_categories(self, column_values, name, dropna):
        global_unique = self.aggregator.fed_union(column_values)
        return pd.Index(global_unique, name=name)

    def _ensure_nan_index(self, idx, column_values):
        has_nan = float(pd.isna(column_values).any())
        global_has_nan = self.aggregator.fed_max(np.array([has_nan]))
        if global_has_nan[0] == 1.0 and not any(pd.isna(idx)):
            return pd.Index(list(idx) + [np.nan], name=idx.name)
        return idx
