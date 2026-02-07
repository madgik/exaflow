from pydantic import BaseModel

from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.statistics.ttest_paired import FederatedTTestPaired

ALGORITHM_NAME = "ttest_paired"


class TTestPairedResult(BaseModel):
    t_stat: float
    df: int
    p: float
    mean_diff: float
    se_diff: float
    ci_upper: str | float
    ci_lower: str | float
    cohens_d: float


class TTestPairedAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self):
        alpha = self.get_parameter("alpha")
        alternative = self.get_parameter("alt_hypothesis")

        results = self.run_local_udf(
            func=local_step,
            kw_args={
                "x_var": self.inputdata.x[0],
                "y_var": self.inputdata.y[0],
                "alpha": alpha,
                "alternative": alternative,
            },
        )
        result = results[0]
        return TTestPairedResult(
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
def local_step(agg_client, data, x_var, y_var, alpha, alternative):
    sample_x = data[x_var].to_numpy(dtype=float, copy=False)
    sample_y = data[y_var].to_numpy(dtype=float, copy=False)

    ttest = FederatedTTestPaired(agg_client=agg_client)
    return ttest.compute(
        sample_x=sample_x,
        sample_y=sample_y,
        alpha=alpha,
        alternative=alternative,
    )
