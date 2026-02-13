from scipy.stats import fisher_exact

from library.statistical_tests.cross_tab.cross_tab import CrossTab
from library.utils.interfaces.statistical_function import StatisticalFunction, NumpyStatisticalFunction


class FisherExact(NumpyStatisticalFunction):

    def compute(self, dataset, *, factor,factor_categories, outcome, outcome_categories):
        cross_tab_table = CrossTab(self.client).compute(dataset, column1=factor,
                                                        column1_categories=factor_categories,
                                                        column2=outcome,
                                                        column2_categories=outcome_categories)
        odds_ratio, p_value = fisher_exact(cross_tab_table)
        return odds_ratio, p_value
