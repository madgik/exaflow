from typing import List
from typing import Optional

from pydantic import BaseModel

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.statistics.anova_twoway import FederatedAnovaTwoWay
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
            desc="Federated two-way (factorial) ANOVA for hypothesis testing with two categorical factors, including main effects and interaction (Type I or II sums of squares).",
            label="Two-way ANOVA (OLS)",
            enabled=True,
            inputdata=specs.InputDataSpecifications(
                y=specs.InputDataSpecification(
                    label="Outcome (dependent)",
                    desc="Single numerical outcome variable.",
                    types=[specs.InputDataType.REAL, specs.InputDataType.INT],
                    stattypes=[specs.InputDataStatType.NUMERICAL],
                    required=True,
                    multiple=False,
                    enumslen=None,
                ),
                x=specs.InputDataSpecification(
                    label="Factors (independent)",
                    desc="Exactly two categorical (nominal) factors.",
                    types=[specs.InputDataType.TEXT],
                    stattypes=[specs.InputDataStatType.NOMINAL],
                    required=True,
                    multiple=True,
                    enumslen=None,
                ),
                validation=None,
            ),
            parameters={
                "sstype": specs.ParameterSpecification(
                    label="Sum of squares type",
                    desc="Sum of squares type: 1 (Type I, sequential) or 2 (Type II, marginal).",
                    types=[specs.ParameterType.INT],
                    required=True,
                    multiple=False,
                    default="2",
                    enums=None,
                    dict_keys_enums=None,
                    dict_values_enums=None,
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

        levels_a = list(self.metadata[x1]["enumerations"])
        levels_b = list(self.metadata[x2]["enumerations"])
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

        results = self.run_local_udf(
            func=anova_twoway_local_step,
            kw_args={
                "x1": x1,
                "x2": x2,
                "y": y,
                "levels_a": levels_a,
                "levels_b": levels_b,
                "sstype": sstype,
            },
        )
        return AnovaResult(**results[0])


@exareme3_udf(with_aggregation_server=True)
def anova_twoway_local_step(agg_client, data, x1, x2, y, levels_a, levels_b, sstype):
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
