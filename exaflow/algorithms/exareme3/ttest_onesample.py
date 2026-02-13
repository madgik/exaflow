from pydantic import BaseModel

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.statistics.ttest_onesample import (
    FederatedTTestOneSample,
)


class TTestOneSampleResult(BaseModel):
    n_obs: int
    std: float
    t_stat: float
    df: int
    p: float
    mean_diff: float
    se_diff: float
    ci_upper: str | float
    ci_lower: str | float
    cohens_d: float


class TTestOneSample(Algorithm):
    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="ttest_onesample",
            desc="Federated Student's one-sample t-test compares the mean of a numeric sample to a specified value (mu). It reports confidence intervals and Cohen's d, with df = n - 1.",
            label="Student's One-Sample T-Test",
            enabled=True,
            inputdata=specs.InputDataSpecifications(
                y=specs.InputDataSpecification(
                    label="Variable",
                    desc="Numeric variable of interest.",
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
                "alt_hypothesis": specs.ParameterSpecification(
                    label="Alternative Hypothesis",
                    desc="Specifies the alternative hypothesis (two-sided, less, or greater).",
                    types=[specs.ParameterType.TEXT],
                    required=True,
                    multiple=False,
                    default="two-sided",
                    enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.LIST,
                        source=["two-sided", "less", "greater"],
                    ),
                    dict_keys_enums=None,
                    dict_values_enums=None,
                    min=None,
                    max=None,
                ),
                "alpha": specs.ParameterSpecification(
                    label="Alpha",
                    desc="The significance level. The probability of rejecting the null hypothesis when it is true.",
                    types=[specs.ParameterType.REAL],
                    required=True,
                    multiple=False,
                    default=0.05,
                    enums=None,
                    dict_keys_enums=None,
                    dict_values_enums=None,
                    min=0.0,
                    max=1.0,
                ),
                "mu": specs.ParameterSpecification(
                    label="Population mean",
                    desc="Null hypothesis mean value (mu0).",
                    types=[specs.ParameterType.REAL],
                    required=True,
                    multiple=False,
                    default=0.0,
                    enums=None,
                    dict_keys_enums=None,
                    dict_values_enums=None,
                    min=-10.0,
                    max=10.0,
                ),
            },
            type=specs.AlgorithmType.EXAREME3,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def run(self):
        alpha = self.get_parameter("alpha")
        alternative = self.get_parameter("alt_hypothesis")
        mu = self.get_parameter("mu")

        results = self.run_local_udf(
            func=local_step,
            kw_args={
                "y_var": self.inputdata.y[0],
                "alpha": alpha,
                "alternative": alternative,
                "mu": mu,
            },
        )
        result = results[0]
        return TTestOneSampleResult(
            n_obs=result["n_obs"],
            std=result["std"],
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
def local_step(agg_client, data, y_var, alpha, alternative, mu):
    sample = data[y_var].to_numpy(dtype=float, copy=False)

    ttest = FederatedTTestOneSample(agg_client=agg_client)
    return ttest.compute(
        sample=sample,
        mu=mu,
        alpha=alpha,
        alternative=alternative,
    )
