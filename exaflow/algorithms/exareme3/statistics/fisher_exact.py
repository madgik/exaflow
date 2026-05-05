from typing import List

from pydantic import BaseModel

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.metadata_enums import get_enum_codes
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.statistics.fisher_exact import (
    FisherExact as FedFisherExact,
)
from exaflow.algorithms.federated.utils.aggregators.numpy_aggregator import (
    NumpyAggregator,
)


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
            desc="v5 - Federated Fisher's Exact test for two categorical variables.",
            label="Fisher's Exact Test",
            enabled=True,
            inputdata=specs.InputDataSpecifications(
                y=specs.InputDataSpecification(
                    label="Outcome (dependent)",
                    desc="Categorical outcome variable.",
                    types=[specs.InputDataType.TEXT],
                    stattypes=[specs.InputDataStatType.NOMINAL],
                    required=True,
                    multiple=False,
                    enumslen=None,
                ),
                x=specs.InputDataSpecification(
                    label="Factor (independent)",
                    desc="Categorical factor variable.",
                    types=[specs.InputDataType.TEXT],
                    stattypes=[specs.InputDataStatType.NOMINAL],
                    required=True,
                    multiple=False,
                    enumslen=None,
                ),
                validation=None,
            ),
            parameters=None,
            type=specs.AlgorithmType.EXAREME3,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def run(self):
        y_var = self.inputdata.y[0]
        x_var = self.inputdata.x[0]

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
    factor_categories = get_enum_codes(metadata, factor)
    outcome_categories = get_enum_codes(metadata, outcome)
    aggregator = NumpyAggregator(agg_client)
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
