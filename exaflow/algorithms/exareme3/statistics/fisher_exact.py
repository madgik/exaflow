from typing import List

from pydantic import BaseModel

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.statistics.fisher_exact import (
    FisherExact as FedFisherExact,
)
from exaflow.algorithms.federated.utils.aggregators.numpy_aggregator import (
    NumpyAggregator,
)
from exaflow.worker_communication import BadUserInput


class FisherExactResult(BaseModel):
    odds_ratio: float
    p_value: float
    x_labels: List[str]
    y_labels: List[str]


class FisherExact(Algorithm):
    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="fisher_exact",
            desc="Fisher's exact test for association between two binary categorical variables.",
            documentation=(
                "Tests association between two categorical variables by "
                "building a 2x2 contingency table and applying Fisher's exact "
                "test. Each selected variable must have exactly two observed "
                "levels after rows with missing values are excluded.\n\n"
                "The result includes the odds ratio, p-value, and category "
                "labels for both variables.\n\n"
                "Reference behavior is aligned with scipy.stats.fisher_exact "
                "for a 2x2 contingency table. The method computes the table "
                "from aggregated category counts without sharing raw data."
            ),
            label="Fisher's Exact Test",
            enabled=True,
            required_preprocessing=["missing_values_handler"],
            y=specs.InputDataSpecification(
                label="Outcome",
                desc="Categorical outcome variable.",
                types=[specs.InputDataType.TEXT],
                stattypes=[specs.InputDataStatType.NOMINAL],
                required=True,
                max_count=1,
            ),
            x=specs.InputDataSpecification(
                label="Factor",
                desc="Categorical factor variable.",
                types=[specs.InputDataType.TEXT],
                stattypes=[specs.InputDataStatType.NOMINAL],
                required=True,
                max_count=1,
            ),
            type=specs.AlgorithmType.EXAREME3,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def run(self):
        y_var = self.y[0]
        x_var = self.x[0]

        results = self.run_local_udf(
            func=fisher_exact_local_step,
            kw_args={
                "factor": x_var,
                "outcome": y_var,
            },
        )

        res = results[0]
        odds_ratio, p_value = res[0], res[1]
        x_labels, y_labels = res[2], res[3]
        return FisherExactResult(
            odds_ratio=odds_ratio,
            p_value=p_value,
            x_labels=x_labels,
            y_labels=y_labels,
        )


@exareme3_udf(with_aggregation_server=True)
def fisher_exact_local_step(agg_client, data, metadata, factor, outcome):
    aggregator = NumpyAggregator(agg_client)
    factor_categories = aggregator.fed_union(data[factor].to_numpy(copy=False)).tolist()
    outcome_categories = aggregator.fed_union(
        data[outcome].to_numpy(copy=False)
    ).tolist()
    if len(factor_categories) != 2 or len(outcome_categories) != 2:
        raise BadUserInput(
            f"Fisher's Exact Test requires exactly 2 observed levels for both "
            f"the factor and the outcome, but factor {factor!r} has "
            f"{len(factor_categories)} and outcome {outcome!r} has "
            f"{len(outcome_categories)}."
        )
    model = FedFisherExact(aggregator=aggregator)
    res = model.compute(
        dataset=data,
        factor=factor,
        outcome=outcome,
        factor_categories=factor_categories,
        outcome_categories=outcome_categories,
        # Missing values are always dropped: NaN rows are excluded from the
        # contingency table. Fisher's Exact Test requires a clean 2x2 table
        # of observed counts with no missing observations.
        dropna=True,
    )
    return (res[0], res[1], res.x_labels, res.y_labels)
