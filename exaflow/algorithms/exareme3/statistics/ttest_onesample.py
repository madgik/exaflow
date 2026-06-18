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
            desc="One-sample t-test comparing a numerical mean with a reference value.",
            documentation=(
                "Compare the mean of a numerical sample to a specified null "
                "hypothesis mean. Degrees of freedom are n - 1 and Cohen's d "
                "is computed relative to the null mean.\n\n"
                "The 'alt_hypothesis' setting selects the alternative hypothesis: "
                "'two-sided', 'less', or 'greater'. Default is 'two-sided'.\n\n"
                "The 'alpha' setting controls the significance level used for "
                "confidence intervals. Default is 0.05.\n\n"
                "The 'mu' setting controls the null hypothesis mean. Default is 0.0.\n\n"
                "The result includes the t statistic, p-value, confidence "
                "interval, sample mean, standard deviation, observation count, "
                "mean difference, standard error of the difference, and "
                "Cohen's d.\n\n"
                "Reference behavior is aligned with scipy.stats one-sample "
                "t-test methodology, with additional confidence interval and "
                "effect-size reporting computed from aggregated sample "
                "statistics without sharing raw data."
            ),
            label="One-sample t-test",
            enabled=True,
            required_preprocessing=["missing_values_handler"],
            y=specs.InputDataSpecification(
                label="Variable",
                desc="Numerical variable to test.",
                types=[specs.InputDataType.REAL, specs.InputDataType.INT],
                stattypes=[specs.InputDataStatType.NUMERICAL],
                required=True,
                max_count=1,
            ),
            parameters={
                "alt_hypothesis": specs.ParameterSpecification(
                    label="Alternative hypothesis",
                    desc="Alternative hypothesis for the mean comparison.",
                    types=[specs.ParameterType.TEXT],
                    required=True,
                    multiple=False,
                    default="two-sided",
                    enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.LIST,
                        source=["two-sided", "less", "greater"],
                    ),
                ),
                "alpha": specs.ParameterSpecification(
                    label="Significance level",
                    desc="Significance level used for confidence intervals.",
                    types=[specs.ParameterType.REAL],
                    required=True,
                    multiple=False,
                    default=0.05,
                    min=0.0,
                    max=1.0,
                ),
                "mu": specs.ParameterSpecification(
                    label="Reference mean",
                    desc="Mean value under the null hypothesis.",
                    types=[specs.ParameterType.REAL],
                    required=True,
                    multiple=False,
                    default=0.0,
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

        result = self.run_local_udf(
            func=local_step,
            kw_args={
                "y_var": self.y[0],
                "alpha": alpha,
                "alternative": alternative,
                "mu": mu,
            },
            identical_results=True,
        )
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
