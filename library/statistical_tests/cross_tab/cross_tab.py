
import pandas as pd
import numpy as np
from library.utils.interfaces.statistical_function import  NumpyStatisticalFunction


class CrossTab(NumpyStatisticalFunction):

    def compute(self, dataset, column1, column2, *, column1_categories = None, column2_categories = None, dropna=False):
        # Convert columns to categorical with the full set of categories
        if column1_categories is None:
            column1_categories = self.aggregator.fed_union(dataset[column1].to_numpy())
        if column2_categories is None:
            column2_categories = self.aggregator.fed_union(dataset[column2].to_numpy())

        # If NaN are not dropped, they should be included in the categories
        if not dropna:
            column1_categories = np.append(column1_categories, np.nan)
            column2_categories = np.append(column2_categories, np.nan)

        # Creating pivot table with zeros:
        zero_table = pd.DataFrame(0, index=column1_categories, columns=column2_categories)

        # Finding Local Cross Tab
        cross_tab = pd.crosstab(dataset[column1],
                                dataset[column2] ,
                                dropna=dropna)

        # Combining local and global crosstabs
        combined = zero_table.add(cross_tab, fill_value=0)

        # Returning the global CrossTab
        combined.iloc[:] = self.aggregator.fed_sum(combined.values)

        if not dropna:
            if (combined.loc[np.nan] == 0).all():
                combined=combined.drop(np.nan, axis=0)
            if (combined[np.nan] == 0).all():
                combined=combined.drop(np.nan, axis=1)
        return combined
