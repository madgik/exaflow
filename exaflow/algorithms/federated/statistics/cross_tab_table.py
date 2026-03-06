
import numpy as np
import pandas as pd

from exaflow.algorithms.federated.utils.aggregators import NumpyAggregator


class CrossTabTable:

    def __init__(self, aggregator: NumpyAggregator):
        self.aggregator = aggregator

    def compute(self, dataset, *, factor,factor_categories, outcome, outcome_categories, dropna=False):
        # Convert columns to categorical with the full set of categories without mutating dataset
        cross_tab = pd.crosstab(dataset[factor], dataset[outcome], dropna=dropna)


        # if not dropna:
        #     factor_categories = list(factor_categories2)
        #     if not any(pd.isna(c) for c in factor_categories):
        #         factor_categories.append(np.nan)
        #
        #     outcome_categories = list(outcome_categories2)
        #     if not any(pd.isna(c) for c in outcome_categories):
        #         outcome_categories.append(np.nan)
        # else:
        #     factor_categories = factor_categories2
        #     outcome_categories = outcome_categories2

        # We must keep dropna=False locally to ensure all clients 
        # submit exactly the same shape to the aggregator.
        # if len(dataset) == 0:
        #     cross_tab = pd.DataFrame(
        #         0,
        #         index=factor_cat.categories,
        #         columns=outcome_cat.categories
        #     )
        # else:
        #     cross_tab = pd.crosstab(factor_cat, outcome_cat, dropna=dropna)
        
        # pd.crosstab may add NaN as a row/column if it exists in the data and dropna=False
        # We must align the local shape exactly to the defined categories

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
            fill_value=0
        )
        # cross_tab = cross_tab.reindex(
        #     index=pd.CategoricalIndex(factor_categories, categories=factor_categories, name=factor),
        #     columns=pd.CategoricalIndex(outcome_categories, categories=outcome_categories, name=outcome),
        #     fill_value=0
        # )
        
        
        cross_tab.iloc[:] = self.aggregator.fed_sum(cross_tab.values)
        

        # Drop rows and columns that are all zeros globally, matching pandas crosstab behavior
        cross_tab = cross_tab.loc[(cross_tab != 0).any(axis=1), (cross_tab != 0).any(axis=0)]
            
        if not dropna:
            expected_factor_idx = pd.Index(factor_categories, name=factor)
            if cross_tab.index.isna().any():
                expected_factor_idx = expected_factor_idx.append(pd.Index([np.nan], name=factor))
            cross_tab.index = expected_factor_idx[expected_factor_idx.isin(cross_tab.index)]

            expected_outcome_idx = pd.Index(outcome_categories, name=outcome)
            if cross_tab.columns.isna().any():
                expected_outcome_idx = expected_outcome_idx.append(pd.Index([np.nan], name=outcome))
            cross_tab.columns = expected_outcome_idx[expected_outcome_idx.isin(cross_tab.columns)]

        return cross_tab
