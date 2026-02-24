from scipy.stats import fisher_exact

from library.statistical_tests.cross_tab.cross_tab import CrossTab
from library.utils.interfaces.statistical_function import  NumpyStatisticalFunction


class FisherExact(NumpyStatisticalFunction):

    def compute(self, dataset, factor, outcome, *, factor_categories=None, outcome_categories=None):
        cross_tab_table = CrossTab(self.aggregator).compute(dataset,
                                                        column1=factor,
                                                        column2=outcome,
                                                        column1_categories=factor_categories,
                                                        column2_categories=outcome_categories,
                                                        dropna=True)
        return fisher_exact(cross_tab_table)
