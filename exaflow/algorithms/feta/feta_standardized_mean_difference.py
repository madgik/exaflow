from pydantic import BaseModel

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.feta import FedColumns
from exaflow.algorithms.feta import FedTensor
from exaflow.algorithms.feta import federated_numpy as fnp
from exaflow.algorithms.feta._smd import pairwise_smd
from exaflow.algorithms.feta.algorithm import Algorithm
from exaflow.worker_communication import BadUserInput


class StandardizedMeanDifferenceComparison(BaseModel):
    group1: str | int
    group2: str | int
    smd: float


class StandardizedMeanDifferenceResult(BaseModel):
    comparisons: list[StandardizedMeanDifferenceComparison]


class FetaStandardizedMeanDifference(Algorithm):
    column_inputs = frozenset({"x"})

    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="feta_standardized_mean_difference",
            desc="Pairwise standardized mean differences for a numerical outcome across observed groups.",
            documentation=(
                "Compute Cohen's d for every unique pair of categories in the "
                "grouping variable. For each category, valid observations determine "
                "the count, mean, and sample variance. Each effect size is the "
                "group1 minus group2 mean difference divided by their pooled sample "
                "standard deviation.\n\n"
                "Categories below the configured privacy minimum are omitted. Missing "
                "values are rejected; preprocessing is not supported. A zero pooled variance produces an effect size of 0.\n\n"
                "The result contains one entry per eligible pair with the category "
                "labels and standardized mean difference. Pair ordering determines "
                "the sign of the effect size.\n\n"
                "The methodology is consistent with the pooled-variance definition "
                "of Cohen's d and is computed from aggregated sufficient statistics "
                "without sharing raw data."
            ),
            label="Standardized Mean Difference",
            enabled=True,
            required_preprocessing=[],
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
            type=specs.AlgorithmType.FETA,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def run(self, x: FedColumns, y: FedTensor) -> StandardizedMeanDifferenceResult:
        name = x.names[0]
        grouping = x[name][:, 0]
        minimum = y.runtime.minimum_group_size
        if minimum is None:
            raise ValueError("FETA worker privacy minimum is unavailable.")
        if grouping.dtype.kind == "U":
            levels = fnp.unique(grouping).tolist()
        else:
            counts = [
                (code, int(fnp.sum(grouping == code)))
                for code in dict.fromkeys(x.metadata[name].categories)
            ]
            if sum(count for _, count in counts) != x.nrows:
                raise BadUserInput(
                    f"Categorical variable {name} contains undeclared codes."
                )
            levels = [code for code, count in counts if count]
        statistics = []
        for code in sorted(levels, key=str):
            sample = y[grouping == code]
            count = len(sample)
            if count < max(2, minimum):
                continue
            mean = float(fnp.sum(sample)) / count
            centered = sample - mean
            variance = float(fnp.sum(centered * centered)) / (count - 1)
            statistics.append((code, count, mean, variance))
        return StandardizedMeanDifferenceResult(
            comparisons=pairwise_smd(statistics, minimum)
        )
