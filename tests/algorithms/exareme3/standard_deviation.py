from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.worker_communication import ColumnDataFloat
from exaflow.worker_communication import ColumnDataStr
from exaflow.worker_communication import TabularDataResult

ALGORITHM_NAME = "standard_deviation"


class StandardDeviationAlgorithm(Algorithm):
    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name=ALGORITHM_NAME,
            desc="Standard Deviation of a column",
            documentation="Standard Deviation of a column",
            label="Standard Deviation",
            enabled=True,
            inputdata=specs.InputDataSpecifications(
                y=specs.InputDataSpecification(
                    label="column",
                    desc="Column",
                    types=[specs.InputDataType.REAL, specs.InputDataType.INT],
                    stattypes=[specs.InputDataStatType.NUMERICAL],
                    required=True,
                    multiple=False,
                    enumslen=None,
                ),
                x=None,
                validation=None,
            ),
            parameters=None,
            type=specs.AlgorithmType.EXAREME3,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def run(self):
        # This call runs the local UDF on all workers.
        # They all execute the aggregation server calls so each one returns the same final standard deviation.
        results = self.run_local_udf(
            func=local_step,
            kw_args={},
        )
        std_deviation = results[0]
        if any(r != std_deviation for r in results[1:]):
            raise ValueError("Worker results do not match")

        # Build the result: for each column in y (here, assume a single column scenario), assign the computed std.
        result = TabularDataResult(
            title="Standard Deviation",
            columns=[
                ColumnDataStr(name="variable", data=self.inputdata.y),
                ColumnDataFloat(name="std_deviation", data=[std_deviation]),
            ],
        )
        return result


def compute_stddev(agg_client, data):
    import numpy as np

    n = len(data)
    if n <= 1:
        return 0.0
    total = agg_client.sum([float(np.sum(data))])[0]
    mean = total / n
    # One remote call instead of two:
    sum_sq = agg_client.sum([float(np.sum(np.square(data)))])[0]
    variance = (sum_sq / n) - mean**2
    return float(np.sqrt(max(variance, 0.0)))


@exareme3_udf(with_aggregation_server=True)
def local_step(agg_client, data, y_var):
    return compute_stddev(agg_client, data[y_var])
