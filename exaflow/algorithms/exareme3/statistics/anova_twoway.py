from typing import List
from typing import Optional

from pydantic import BaseModel

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.statistics.anova_twoway import FederatedAnovaTwoWay
from exaflow.algorithms.federated.utils.aggregators.numpy_aggregator import (
    NumpyAggregator,
)
from exaflow.worker_communication import BadUserInput


class AnovaResult(BaseModel):
    terms: List[str]
    sum_sq: List[float]
    df: List[int]
    f_stat: List[Optional[float]]
    f_pvalue: List[Optional[float]]


class AnovaTwoWay(Algorithm):
    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="anova_twoway",
            desc="Two-way ANOVA with two categorical factors.",
            documentation=(
                "Test whether a numerical outcome differs across combinations "
                "of two categorical factors. The model includes main effects "
                "and their interaction.\n\n"
                "The 'sstype' setting selects the sums-of-squares type:\n"
                "  - 1 computes Type I sequential sums of squares.\n"
                "  - 2 computes Type II marginal sums of squares. Default is 2.\n\n"
                "The result includes an ANOVA table for the fitted factorial model."
            ),
            label="Two-way ANOVA (OLS)",
            enabled=True,
            required_preprocessing=["missing_values_handler"],
            inputdata=specs.InputDataSpecifications(
                y=specs.InputDataSpecification(
                    label="Outcome (dependent)",
                    desc="Single numerical outcome variable.",
                    types=[specs.InputDataType.REAL, specs.InputDataType.INT],
                    stattypes=[specs.InputDataStatType.NUMERICAL],
                    required=True,
                    max_count=1,
                ),
                x=specs.InputDataSpecification(
                    label="Factors (independent)",
                    desc="Exactly two categorical (nominal) factors.",
                    types=[specs.InputDataType.TEXT],
                    stattypes=[specs.InputDataStatType.NOMINAL],
                    required=True,
                    min_count=2,
                    max_count=2,
                ),
            ),
            parameters={
                "sstype": specs.ParameterSpecification(
                    label="Sum of squares type",
                    desc="Sums-of-squares method for the ANOVA table.",
                    types=[specs.ParameterType.INT],
                    required=True,
                    multiple=False,
                    default="2",
                    min=1,
                    max=2,
                ),
            },
            type=specs.AlgorithmType.EXAREME3,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def run(self):
        y = self.inputdata.y[0]
        xs = self.inputdata.x
        if len(xs) != 2:
            raise BadUserInput("ANOVA two-way requires exactly two covariates (x).")
        x1, x2 = xs

        sstype = self.get_parameter("sstype")

        result = self.run_local_udf(
            func=local_step,
            kw_args={
                "x1": x1,
                "x2": x2,
                "y": y,
                "sstype": sstype,
            },
            identical_results=True,
        )
        return AnovaResult(**result)


@exareme3_udf(with_aggregation_server=True)
def local_step(agg_client, data, x1, x2, y, sstype):
    aggregator = NumpyAggregator(agg_client)
    levels_a = aggregator.fed_union(data[x1].to_numpy(copy=False)).tolist()
    levels_b = aggregator.fed_union(data[x2].to_numpy(copy=False)).tolist()

    if len(levels_a) < 2:
        raise BadUserInput(
            f"The variable {x1} has less than 2 levels and Anova cannot be "
            "performed. Please choose another variable."
        )
    if len(levels_b) < 2:
        raise BadUserInput(
            f"The variable {x2} has less than 2 levels and Anova cannot be "
            "performed. Please choose another variable."
        )

    model = FederatedAnovaTwoWay(agg_client=agg_client, sstype=sstype)
    model.fit(
        data=data,
        y=y,
        x1=x1,
        x2=x2,
        levels_a=levels_a,
        levels_b=levels_b,
    )

    terms = model.terms_
    return {
        "terms": terms,
        "sum_sq": [model.sum_sq_[term] for term in terms],
        "df": [model.df_[term] for term in terms],
        "f_stat": [model.f_stat_[term] for term in terms],
        "f_pvalue": [model.f_pvalue_[term] for term in terms],
    }
