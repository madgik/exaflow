from pydantic import BaseModel

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.statistics.standardized_mean_difference import (
    FederatedStandardizedMeanDifference,
)


class StandardizedMeanDifferenceComparison(BaseModel):
    group1: str | int
    group2: str | int
    smd: float


class StandardizedMeanDifferenceResult(BaseModel):
    comparisons: list[StandardizedMeanDifferenceComparison]


class StandardizedMeanDifference(Algorithm):
    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="standardized_mean_difference",
            desc="Pairwise standardized mean differences for a numerical outcome across observed groups.",
            documentation=(
                "Compute Cohen's d for every unique pair of categories in the "
                "grouping variable. For each category, valid observations determine "
                "the count, mean, and sample variance. Each effect size is the "
                "group1 minus group2 mean difference divided by their pooled sample "
                "standard deviation.\n\n"
                "Categories below the configured privacy minimum are omitted. Missing "
                "group labels and missing outcome values do not contribute to the "
                "calculation. A zero pooled variance produces an effect size of 0.\n\n"
                "The result contains one entry per eligible pair with the category "
                "labels and standardized mean difference. Pair ordering determines "
                "the sign of the effect size.\n\n"
                "The methodology is consistent with the pooled-variance definition "
                "of Cohen's d and is computed from aggregated sufficient statistics "
                "without sharing raw data."
            ),
            label="Standardized Mean Difference",
            enabled=True,
            required_preprocessing=["missing_values_handler"],
            y=specs.InputDataSpecification(
                label="Outcome",
                desc="Numerical variable whose group means are compared.",
                types=[specs.InputDataType.REAL, specs.InputDataType.INT],
                stattypes=[specs.InputDataStatType.NUMERICAL],
                required=True,
                max_count=1,
            ),
            x=specs.InputDataSpecification(
                label="Grouping variable",
                desc="Categorical variable defining the groups compared pairwise.",
                types=[specs.InputDataType.TEXT, specs.InputDataType.INT],
                stattypes=[specs.InputDataStatType.NOMINAL],
                required=True,
                max_count=1,
            ),
            parameters={},
            type=specs.AlgorithmType.EXAREME3,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def run(self):
        comparisons = self.run_local_udf(
            func=local_step,
            kw_args={
                "group_var": self.x[0],
                "value_var": self.y[0],
            },
            identical_results=True,
        )
        return StandardizedMeanDifferenceResult(comparisons=comparisons)


@exareme3_udf(with_aggregation_server=True)
def local_step(agg_client, data, group_var, value_var):
    from exaflow.worker import config as worker_config

    estimator = FederatedStandardizedMeanDifference(agg_client=agg_client)
    result = estimator.pairwise_by_group(
        data=data,
        value_var=value_var,
        group_var=group_var,
        minimum_group_size=worker_config.privacy.minimum_row_count,
    )
    comparisons = result[["group1", "group2", "smd"]].to_dict(orient="records")
    for comparison in comparisons:
        for group_field in ("group1", "group2"):
            value = comparison[group_field]
            comparison[group_field] = value.item() if hasattr(value, "item") else value
        comparison["smd"] = float(comparison["smd"])
    return comparisons
