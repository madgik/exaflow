from scipy.stats import fisher_exact

from exaflow.algorithms.federated.statistics.cross_tab_table import CrossTabTable
from exaflow.algorithms.federated.utils.aggregators import NumpyAggregator


class FisherExact:

    def __init__(self, aggregator: NumpyAggregator):
        self.aggregator = aggregator

    def compute(self, dataset, *, factor, factor_categories, outcome, outcome_categories, dropna=False):
        cross_tab_table = CrossTabTable(self.aggregator).compute(dataset, factor=factor,
                                                                factor_categories=factor_categories,
                                                                outcome=outcome,
                                                                outcome_categories=outcome_categories,
                                                                dropna=dropna)
        odds_ratio, p_value = fisher_exact(cross_tab_table)
        return odds_ratio, p_value
