from pydantic import BaseModel

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.statistics.ttest_paired import FederatedTTestPaired


class TTestPairedResult(BaseModel):
    t_stat: float
    df: int
    p: float
    mean_diff: float
    se_diff: float
    ci_upper: str | float
    ci_lower: str | float
    cohens_d: float


class TTestPaired(Algorithm):
    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="ttest_paired",
            desc="Paired-samples t-test comparing two related numerical measurements.",
            documentation=(
                "Compare the mean of paired differences between two related "
                "numerical measurements. Degrees of freedom are n - 1, and "
                "Cohen's d is computed from the paired differences.\n\n"
                "The 'alt_hypothesis' setting selects the alternative hypothesis: "
                "'two-sided', 'less', or 'greater'. Default is 'two-sided'.\n\n"
                "The 'alpha' setting controls the significance level used for "
                "confidence intervals. Default is 0.05.\n\n"
                "The result includes the t statistic, p-value, confidence "
                "interval, mean difference, standard error of the difference, "
                "degrees of freedom, and Cohen's d.\n\n"
                "Reference behavior is aligned with scipy.stats.ttest_rel "
                "paired t-test methodology, with additional confidence "
                "interval and effect-size reporting computed from aggregated "
                "paired-difference statistics without sharing raw data."
            ),
            label="Paired t-test",
            enabled=True,
            required_preprocessing=["missing_values_handler"],
            y=specs.InputDataSpecification(
                label="Measurement 1",
                desc="First numerical measurement in each pair.",
                types=[specs.InputDataType.REAL, specs.InputDataType.INT],
                stattypes=[specs.InputDataStatType.NUMERICAL],
                required=True,
                max_count=1,
            ),
            x=specs.InputDataSpecification(
                label="Measurement 2",
                desc="Second numerical measurement in each pair.",
                types=[specs.InputDataType.REAL, specs.InputDataType.INT],
                stattypes=[specs.InputDataStatType.NUMERICAL],
                required=True,
                max_count=1,
            ),
            parameters={
                "alt_hypothesis": specs.ParameterSpecification(
                    label="Alternative hypothesis",
                    desc="Alternative hypothesis for the paired difference.",
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
            },
            type=specs.AlgorithmType.EXAREME3,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def run(self):
        alpha = self.get_parameter("alpha")
        alternative = self.get_parameter("alt_hypothesis")

        result = self.run_local_udf(
            func=local_step,
            kw_args={
                "x_var": self.x[0],
                "y_var": self.y[0],
                "alpha": alpha,
                "alternative": alternative,
            },
            identical_results=True,
        )
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
