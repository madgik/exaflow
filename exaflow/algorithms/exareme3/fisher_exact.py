from pydantic import BaseModel

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from library.statistical_tests.nonparametric.fisher_exact import FisherExact as LibFisherExact
from library.utils.aggregators.numpy_aggregator import NumpyAggregator


class FisherExactResult(BaseModel):
    title: str
    odds_ratio: float
    p_value: float


class FisherExact(Algorithm):
    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="fisher_exact",
            desc="Federated Fisher's exact test on a 2x2 contingency table.",
            label="Fisher's Exact Test",
            enabled=True,
            inputdata=specs.InputDataSpecifications(
                y=specs.InputDataSpecification(
                    label="Factor 1",
                    desc="First nominal variable (must have 2 levels).",
                    types=[specs.InputDataType.TEXT, specs.InputDataType.INT],
                    stattypes=[specs.InputDataStatType.NOMINAL],
                    required=True,
                    multiple=False,
                    enumslen=None,
                ),
                x=specs.InputDataSpecification(
                    label="Factor 2",
                    desc="Second nominal variable (must have 2 levels).",
                    types=[specs.InputDataType.TEXT, specs.InputDataType.INT],
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
        results = self.run_local_udf(
            func=local_step,
            kw_args={
                "factor": self.inputdata.y[0],
                "outcome": self.inputdata.x[0],
            },
        )
        odds_ratio, p_value = results[0]

        return FisherExactResult(
            title="Fisher's Exact Test",
            odds_ratio=odds_ratio,
            p_value=p_value,
        )


@exareme3_udf(with_aggregation_server=True)
def local_step(agg_client, data, factor, outcome):
    aggregator = NumpyAggregator(agg_client)
    fe = LibFisherExact(aggregator)
    return fe.compute(
        dataset=data,
        factor=factor,
        outcome=outcome,
    )
