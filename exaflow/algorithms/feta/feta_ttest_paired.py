import numpy as np
import scipy.stats as st
from pydantic import BaseModel

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.feta import FedTensor
from exaflow.algorithms.feta import federated_numpy as fnp
from exaflow.algorithms.feta.algorithm import Algorithm
from exaflow.algorithms.specifications import AlgorithmType
from exaflow.worker_communication import BadUserInput


class TTestPairedResult(BaseModel):
    t_stat: float
    df: int
    p: float
    mean_diff: float
    se_diff: float
    ci_upper: str | float
    ci_lower: str | float
    cohens_d: float


class FetaTTestPaired(Algorithm):
    @classmethod
    def get_specification(cls):
        specification = specs.AlgorithmSpecification(
            name="feta_ttest_paired",
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
            required_preprocessing=[],
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
            type=specs.AlgorithmType.FETA,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )
        return specification.model_copy(
            update={
                "name": "feta_ttest_paired",
                "type": AlgorithmType.FETA,
                "required_preprocessing": [],
                "documentation": specification.documentation
                + (
                    "\n\nDifferences are computed as Measurement 2 minus "
                    "Measurement 1 (x minus y). Inputs containing missing "
                    "values are rejected. Preprocessing is not supported."
                ),
            }
        )

    def run(self, x: FedTensor, y: FedTensor) -> TTestPairedResult:
        if x.shape != y.shape:
            raise BadUserInput("Paired samples must have the same length.")
        differences = x - y
        n = float(len(x))
        sum_x = fnp.sum(x)
        sum_y = fnp.sum(y)
        sum_diff = fnp.sum(differences)
        sum_squared_diff = fnp.sum(differences * differences)
        if n <= 1:
            raise BadUserInput("Not enough observations for paired t-test.")

        alpha = self.parameters.get("alpha", 0.05)
        alternative = self.parameters.get("alt_hypothesis", "two-sided")
        mean_x = sum_x / n
        mean_y = sum_y / n
        mean_diff = sum_diff / n
        std_diff = np.sqrt((sum_squared_diff - sum_diff**2 / n) / (n - 1))
        se = std_diff / np.sqrt(n)
        t_stat = (mean_x - mean_y) / se
        df = n - 1
        lower, upper = st.t.interval(1 - alpha, df, loc=mean_diff, scale=se)
        if alternative == "greater":
            p = 1.0 - st.t.cdf(t_stat, df)
            upper = "Infinity"
        elif alternative == "less":
            p = 1.0 - st.t.cdf(-t_stat, df)
            lower = "-Infinity"
        else:
            p = (1.0 - st.t.cdf(abs(t_stat), df)) * 2.0

        return TTestPairedResult(
            t_stat=t_stat,
            df=int(df),
            p=p,
            mean_diff=mean_diff,
            se_diff=se,
            ci_upper=upper,
            ci_lower=lower,
            cohens_d=mean_diff / std_diff,
        )
