from typing import List

from pydantic import BaseModel

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from library.statistical_tests.parametric.chi_squared import ChiSquared as LibChiSquared
from library.utils.aggregators.numpy_aggregator import NumpyAggregator


class ChiSquaredResult(BaseModel):
    title: str
    chi2: float
    p_value: float
    dof: int
    expected: List[List[float]]


class ChiSquared(Algorithm):
    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="chi_squared",
            desc="Federated Chi-squared test of independence of variables in a contingency table.",
            label="Chi-Squared Test",
            enabled=True,
            inputdata=specs.InputDataSpecifications(
                y=specs.InputDataSpecification(
                    label="Factor 1",
                    desc="First nominal variable.",
                    types=[specs.InputDataType.TEXT, specs.InputDataType.INT],
                    stattypes=[specs.InputDataStatType.NOMINAL],
                    required=True,
                    multiple=False,
                    enumslen=None,
                ),
                x=specs.InputDataSpecification(
                    label="Factor 2",
                    desc="Second nominal variable.",
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
        chi2, p, dof, expected = results[0]

        return ChiSquaredResult(
            title="Chi-Squared Test",
            chi2=chi2,
            p_value=p,
            dof=dof,
            expected=expected.tolist(),
        )


@exareme3_udf(with_aggregation_server=True)
def local_step(agg_client, data, factor, outcome):
    aggregator = NumpyAggregator(agg_client)
    cs = LibChiSquared(aggregator)
    return cs.compute(
        dataset=data,
        factor=factor,
        outcome=outcome,
    )
