from typing import List
from typing import Optional

from pydantic import BaseModel

from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.anova_twoway import FederatedAnovaTwoWay
from exaflow.worker_communication import BadUserInput

ALGORITHM_NAME = "anova_twoway"


class AnovaResult(BaseModel):
    terms: List[str]
    sum_sq: List[float]
    df: List[int]
    f_stat: List[Optional[float]]
    f_pvalue: List[Optional[float]]


class AnovaTwoWayAlgorithm(Algorithm, algname=ALGORITHM_NAME):
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
    try:
        model.fit(
            data=data,
            y=y,
            x1=x1,
            x2=x2,
            levels_a=levels_a,
            levels_b=levels_b,
        )
    except ValueError as exc:
        raise BadUserInput(str(exc))

    terms = model.terms_
    return {
        "terms": terms,
        "sum_sq": [model.sum_sq_[term] for term in terms],
        "df": [model.df_[term] for term in terms],
        "f_stat": [model.f_stat_[term] for term in terms],
        "f_pvalue": [model.f_pvalue_[term] for term in terms],
    }
