from typing import List
from typing import Optional

from pydantic import BaseModel

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from library.statistical_tests.cross_tab.cross_tab import CrossTab as LibCrossTab
from library.utils.aggregators.numpy_aggregator import NumpyAggregator


class CrossTabResult(BaseModel):
    title: str
    columns: List[str]
    index: List[str]
    data: List[List[int]]


class CrossTab(Algorithm):
    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="cross_tab",
            desc="Federated contingency table (cross-tabulation) of two variables.",
            label="Cross Tabulation",
            enabled=True,
            inputdata=specs.InputDataSpecifications(
                y=specs.InputDataSpecification(
                    label="Row Variable",
                    desc="Variable to use for the rows of the contingency table.",
                    types=[specs.InputDataType.TEXT, specs.InputDataType.INT],
                    stattypes=[specs.InputDataStatType.NOMINAL],
                    required=True,
                    multiple=False,
                    enumslen=None,
                ),
                x=specs.InputDataSpecification(
                    label="Column Variable",
                    desc="Variable to use for the columns of the contingency table.",
                    types=[specs.InputDataType.TEXT, specs.InputDataType.INT],
                    stattypes=[specs.InputDataStatType.NOMINAL],
                    required=True,
                    multiple=False,
                    enumslen=None,
                ),
                validation=None,
            ),
            parameters={
                "dropna": specs.ParameterSpecification(
                    label="Drop NA",
                    desc="If True, do not include counts of NaN/NA values.",
                    types=[specs.ParameterType.BOOLEAN],
                    required=False,
                    multiple=False,
                    default=False,
                    enums=None,
                    dict_keys_enums=None,
                    dict_values_enums=None,
                    min=None,
                    max=None,
                ),
            },
            type=specs.AlgorithmType.EXAREME3,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def run(self):
        dropna = self.get_parameter("dropna", False)

        results = self.run_local_udf(
            func=local_step,
            kw_args={
                "column1": self.inputdata.y[0],
                "column2": self.inputdata.x[0],
                "dropna": dropna,
            },
        )
        combined = results[0]

        return CrossTabResult(
            title="Cross Tabulation",
            columns=combined.columns.tolist(),
            index=combined.index.tolist(),
            data=combined.values.tolist(),
        )


@exareme3_udf(with_aggregation_server=True)
def local_step(agg_client, data, column1, column2, dropna):
    aggregator = NumpyAggregator(agg_client)
    ct = LibCrossTab(aggregator)
    return ct.compute(
        dataset=data,
        column1=column1,
        column2=column2,
        dropna=dropna,
    )
