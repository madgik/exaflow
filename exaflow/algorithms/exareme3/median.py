from pydantic import BaseModel

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from library.descriptive_stats.median import MedianBasedOnHistogram as LibMedian
from library.utils.aggregators.numpy_aggregator import NumpyAggregator


class MedianResult(BaseModel):
    title: str
    median: float


class Median(Algorithm):
    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="median",
            desc="Federated median estimation based on histogram binning.",
            label="Median",
            enabled=True,
            inputdata=specs.InputDataSpecifications(
                y=specs.InputDataSpecification(
                    label="Variable",
                    desc="Numeric variable to compute median for.",
                    types=[specs.InputDataType.REAL, specs.InputDataType.INT],
                    stattypes=[specs.InputDataStatType.NUMERICAL],
                    required=True,
                    multiple=False,
                    enumslen=None,
                ),
                x=None,
                validation=None,
            ),
            parameters={
                "range_acc": specs.ParameterSpecification(
                    label="Accuracy",
                    desc="Range accuracy for histogram binning (inverse of bin count).",
                    types=[specs.ParameterType.REAL],
                    required=False,
                    multiple=False,
                    default=0.1,
                    enums=None,
                    dict_keys_enums=None,
                    dict_values_enums=None,
                    min=0.01,
                    max=1.0,
                ),
            },
            type=specs.AlgorithmType.EXAREME3,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def run(self):
        range_acc = self.get_parameter("range_acc", 0.1)

        results = self.run_local_udf(
            func=local_step,
            kw_args={
                "y_var": self.inputdata.y[0],
                "range_acc": range_acc,
            },
        )
        median = results[0]

        return MedianResult(
            title="Median",
            median=median,
        )


@exareme3_udf(with_aggregation_server=True)
def local_step(agg_client, data, y_var, range_acc):
    aggregator = NumpyAggregator(agg_client)
    x = data[y_var].to_numpy()

    m = LibMedian(aggregator)
    return m.compute(
        x=x,
        range_acc=range_acc,
    )
