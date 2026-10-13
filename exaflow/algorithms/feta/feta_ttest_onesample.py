import numpy as np
import scipy.stats as st
from pydantic import BaseModel

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.feta import FedTensor
from exaflow.algorithms.feta import federated_numpy as fnp
from exaflow.algorithms.feta.algorithm import Algorithm
from exaflow.algorithms.specifications import AlgorithmType
from exaflow.worker_communication import BadUserInput


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


class FetaTTestOneSample(Algorithm):
    @classmethod
    def get_specification(cls):
        specification = specs.AlgorithmSpecification(
            name="feta_ttest_onesample",
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
            required_preprocessing=[],
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
            type=specs.AlgorithmType.FETA,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )
        return specification.model_copy(
            update={
                "name": "feta_ttest_onesample",
                "type": AlgorithmType.FETA,
                "required_preprocessing": [],
                "documentation": specification.documentation
                + (
                    "\n\nInputs containing missing values are rejected. "
                    "Preprocessing is not supported."
                ),
            }
        )

    def run(self, y: FedTensor) -> TTestOneSampleResult:
        n = float(len(y))
        sum_y = fnp.sum(y)
        sum_squared_y = fnp.sum(y * y)
        if n <= 1:
            raise BadUserInput("Not enough observations for one-sample t-test.")

        mu = self.parameters.get("mu", 0.0)
        alpha = self.parameters.get("alpha", 0.05)
        alternative = self.parameters.get("alt_hypothesis", "two-sided")
        diff = sum_y - n * mu
        diff_squared = sum_squared_y - 2 * mu * sum_y + n * mu**2
        mean = sum_y / n
        std = np.sqrt((diff_squared - diff**2 / n) / (n - 1))
        se = std / np.sqrt(n)
        t_stat = (mean - mu) / se
        df = n - 1
        lower, upper = st.t.interval(1 - alpha, df, loc=mean, scale=se)
        if alternative == "greater":
            p = 1.0 - st.t.cdf(t_stat, df)
            upper = "Infinity"
        elif alternative == "less":
            p = 1.0 - st.t.cdf(-t_stat, df)
            lower = "-Infinity"
        else:
            p = (1.0 - st.t.cdf(abs(t_stat), df)) * 2.0

        return TTestOneSampleResult(
            n_obs=int(n),
            std=std,
            t_stat=t_stat,
            df=int(df),
            p=p,
            mean_diff=mean,
            se_diff=se,
            ci_upper=upper,
            ci_lower=lower,
            cohens_d=(mean - mu) / std,
        )
