from pydantic import BaseModel

from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.statistics.ttest_independent import (
    FederatedTTestIndependent,
)
from exaflow.algorithms.specifications import AlgorithmName


class TTestIndependentResult(BaseModel):
    t_stat: float
    df: float
    p: float
    mean_diff: float
    se_diff: float
    ci_upper: str | float
    ci_lower: str | float
    cohens_d: float


class TTestIndependent(Algorithm, algname=AlgorithmName.TTEST_INDEPENDENT):
    def run(self):
        alpha = self.get_parameter("alpha")
        alternative = self.get_parameter("alt_hypothesis")
        group_a = self.get_parameter("groupA")
        group_b = self.get_parameter("groupB")

        results = self.run_local_udf(
            func=local_step,
            kw_args={
                "group_var": self.inputdata.x[0],
                "value_var": self.inputdata.y[0],
                "alpha": alpha,
                "alternative": alternative,
                "group_a": group_a,
                "group_b": group_b,
            },
        )

        result = results[0]
        return TTestIndependentResult(
            t_stat=result["t_stat"],
            df=result["df"],
            p=result["p_value"],
            mean_diff=result["mean_diff"],
            se_diff=result["se_diff"],
            ci_upper=result["ci_upper"],
            ci_lower=result["ci_lower"],
            cohens_d=result["cohens_d"],
        )


@exareme3_udf(with_aggregation_server=True)
def local_step(
    agg_client, data, group_var, value_var, alpha, alternative, group_a, group_b
):
    ttest = FederatedTTestIndependent(agg_client=agg_client)
    return ttest.compute(
        data=data,
        group_var=group_var,
        value_var=value_var,
        group_a=group_a,
        group_b=group_b,
        alpha=alpha,
        alternative=alternative,
    )
