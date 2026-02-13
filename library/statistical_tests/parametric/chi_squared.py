from library.statistical_tests.cross_tab.cross_tab import CrossTab
from library.utils.interfaces.statistical_function import StatisticalFunction
from scipy.stats import chi2_contingency


class ChiSquared(StatisticalFunction):

    def compute(self, dataset, *, factor,factor_categories, outcome, outcome_categories):
        cross_tab_table = CrossTab(self.client).compute(dataset, column1=factor,
                                                        column1_categories=factor_categories,
                                                        column2=outcome,
                                                        column2_categories=outcome_categories)
        chi2, p, dof, expected = chi2_contingency(cross_tab_table)
        return chi2, p, dof, expected
