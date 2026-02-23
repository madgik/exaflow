from library.statistical_tests.cross_tab.cross_tab import CrossTab
from library.utils.interfaces.statistical_function import NumpyStatisticalFunction
from scipy.stats import chi2_contingency


class ChiSquared(NumpyStatisticalFunction):

    def compute(self, dataset, factor, outcome, *, factor_categories=None, outcome_categories=None):
        cross_tab_table = CrossTab(self.aggregator).compute(dataset,
                                                        column1=factor,
                                                        column2=outcome,
                                                        column1_categories=factor_categories,
                                                        column2_categories=outcome_categories,
                                                        dropna=True)
        chi2, p, dof, expected = chi2_contingency(cross_tab_table)
        return chi2, p, dof, expected
