from typing import List
from typing import Optional

from pydantic import BaseModel

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.statistics.percentile import Percentile
from exaflow.algorithms.federated.utils.aggregators.numpy_aggregator import (
    NumpyAggregator,
)

_QUARTILE_VALUES = [0.25, 0.5, 0.75]


class QuartileResult(BaseModel):
    q: float
    value: Optional[float]
    actual_q: Optional[float]


class QuartilesResult(BaseModel):
    quantiles: List[QuartileResult]


class QuartilesAlgorithm(Algorithm):
    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="quartiles",
            desc="Estimate the three quartiles (Q1=0.25, Q2=0.50, Q3=0.75) of a numerical variable via iterative histogram refinement.",
            documentation=(
                "Estimates Q1 (25th percentile), Q2 (median), and Q3 (75th percentile) "
                "of a numerical variable without sharing raw data.\n\n"
                "For each quartile the procedure builds a histogram over the data range "
                "and refines the estimate by re-binning the bucket that contains the "
                "target rank, for up to 'max_iterations' passes. Integer variables use "
                "whole-number bin edges; real variables use equal-width bins.\n\n"
                "Each result entry also returns 'actual_q' — the cumulative fraction of "
                "data at or below the returned value. It is null when the value falls "
                "exactly on the requested quartile, and otherwise reports the achieved "
                "fraction, which can exceed the target when the variable has repeated "
                "values.\n\n"
                "Returns a null value and actual_q when no finite data is available."
            ),
            label="Quartiles (Histogram-Based)",
            enabled=True,
            y=specs.InputDataSpecification(
                label="Variable",
                desc="Numerical variable to estimate the quartiles for.",
                types=[specs.InputDataType.REAL, specs.InputDataType.INT],
                stattypes=[specs.InputDataStatType.NUMERICAL],
                required=True,
                max_count=1,
            ),
            parameters={
                "num_bins": specs.ParameterSpecification(
                    label="Number of bins",
                    desc="Bin count used for each histogram refinement step.",
                    types=[specs.ParameterType.INT],
                    required=False,
                    multiple=False,
                    default=20,
                    min=2,
                    max=100,
                ),
            },
            type=specs.AlgorithmType.EXAREME3,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def run(self):
        num_bins = int(self.get_parameter("num_bins", 20) or 20)
        y_var = self.y[0]
        is_integer = (
            self.metadata.get(y_var, {}).get("sql_type")
            == specs.InputDataType.INT.value
        )

        result = self.run_local_udf(
            func=local_step,
            kw_args={
                "y_var": y_var,
                "q_values": _QUARTILE_VALUES,
                "num_bins": num_bins,
                "is_integer": is_integer,
            },
            identical_results=True,
        )
        return QuartilesResult(
            quantiles=[QuartileResult(**item) for item in result["quantiles"]]
        )


@exareme3_udf(with_aggregation_server=True)
def local_step(agg_client, data, y_var, q_values, num_bins, is_integer):
    aggregator = NumpyAggregator(agg_client)
    estimator = Percentile(aggregator)
    quantiles = []
    for q in q_values:
        result = estimator.compute(
            data[y_var],
            float(q),
            num_bins=int(num_bins),
            is_integer=bool(is_integer),
        )
        if result is None:
            quantiles.append({"q": float(q), "value": None, "actual_q": None})
        else:
            value, actual_q = result
            quantiles.append(
                {
                    "q": float(q),
                    "value": float(value) if value is not None else None,
                    "actual_q": float(actual_q) if actual_q is not None else None,
                }
            )
    return {"quantiles": quantiles}
