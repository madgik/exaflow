import numpy as np
import scipy.stats as st
from pydantic import BaseModel

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.feta import FedTensor
from exaflow.algorithms.feta import federated_numpy as fnp
from exaflow.algorithms.feta.algorithm import Algorithm
from exaflow.algorithms.specifications import AlgorithmType
from exaflow.worker_communication import BadUserInput


class TTestIndependentResult(BaseModel):
    t_stat: float
    df: float
    p: float
    mean_diff: float
    se_diff: float
    ci_upper: str | float
    ci_lower: str | float
    cohens_d: float


class FetaTTestIndependent(Algorithm):
    @classmethod
    def get_specification(cls):
        specification = specs.AlgorithmSpecification(
            name="feta_ttest_independent",
            desc="Independent-samples t-test comparing a numerical variable across two groups.",
            documentation=(
                "Compare the means of a numerical variable between two "
                "independent groups using pooled variance. Degrees of freedom "
                "are n_a + n_b - 2, and Cohen's d is computed from the pooled "
                "standard deviation.\n\n"
                "The 'alt_hypothesis' setting selects whether group A is "
                "different from, less than, or greater than group B. Default is "
                "'two-sided'.\n\n"
                "The 'alpha' setting controls the significance level used for "
                "confidence intervals. Default is 0.05.\n\n"
                "The 'groupA' and 'groupB' settings select the grouping-variable "
                "categories compared by the test.\n\n"
                "The result includes the t statistic, p-value, confidence "
                "interval, group means, group sizes, degrees of freedom, and "
                "Cohen's d.\n\n"
                "Reference behavior is aligned with the equal-variance "
                "independent t-test methodology used by scipy.stats.ttest_ind "
                "with equal_var=True. The method computes the test quantities "
                "from aggregated group statistics without sharing raw data."
            ),
            label="Independent t-test",
            enabled=True,
            required_preprocessing=[],
            y=specs.InputDataSpecification(
                label="Outcome",
                desc="Numerical variable compared between groups.",
                types=[specs.InputDataType.REAL, specs.InputDataType.INT],
                stattypes=[specs.InputDataStatType.NUMERICAL],
                required=True,
                max_count=1,
            ),
            x=specs.InputDataSpecification(
                label="Grouping variable",
                desc="Categorical variable containing the two groups.",
                types=[specs.InputDataType.TEXT],
                stattypes=[specs.InputDataStatType.NOMINAL],
                required=True,
                max_count=1,
            ),
            parameters={
                "alt_hypothesis": specs.ParameterSpecification(
                    label="Alternative hypothesis",
                    desc="Alternative hypothesis for the group comparison.",
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
                "groupA": specs.ParameterSpecification(
                    label="Group A",
                    desc="Grouping-variable category used as group A.",
                    types=[specs.ParameterType.TEXT, specs.ParameterType.INT],
                    required=True,
                    multiple=False,
                    enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.INPUT_VAR_CDE_ENUMS,
                        source=["x"],
                    ),
                ),
                "groupB": specs.ParameterSpecification(
                    label="Group B",
                    desc="Grouping-variable category used as group B.",
                    types=[specs.ParameterType.TEXT, specs.ParameterType.INT],
                    required=True,
                    multiple=False,
                    enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.INPUT_VAR_CDE_ENUMS,
                        source=["x"],
                    ),
                ),
            },
            type=specs.AlgorithmType.FETA,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )
        return specification.model_copy(
            update={
                "name": "feta_ttest_independent",
                "type": AlgorithmType.FETA,
                "required_preprocessing": [],
                "documentation": specification.documentation
                + "\n\nInputs containing missing values are rejected. Preprocessing is not supported.",
            }
        )

    def run(self, x: FedTensor, y: FedTensor) -> TTestIndependentResult:
        grouping = x[:, 0]
        sample_a = y[grouping == self.parameters["groupA"]]
        sample_b = y[grouping == self.parameters["groupB"]]
        n_a, n_b = float(len(sample_a)), float(len(sample_b))
        sum_a, sum_b = fnp.sum(sample_a), fnp.sum(sample_b)
        sq_a = fnp.sum(sample_a * sample_a)
        sq_b = fnp.sum(sample_b * sample_b)
        if n_a < 1:
            raise BadUserInput("Group A has no data.")
        if n_b < 1:
            raise BadUserInput("Group B has no data.")
        if n_a + n_b <= 2:
            raise BadUserInput("Not enough observations for independent t-test.")
        mean_a, mean_b = sum_a / n_a, sum_b / n_b
        # Preserve the reference's singleton-group and constant-sample behavior.
        sd_a = np.sqrt(
            (sq_a - 2 * mean_a * sum_a + mean_a**2 * n_a) / np.float64(n_a - 1)
        )
        sd_b = np.sqrt(
            (sq_b - 2 * mean_b * sum_b + mean_b**2 * n_b) / np.float64(n_b - 1)
        )
        df = n_a + n_b - 2
        pooled = ((n_a - 1) * sd_a**2 + (n_b - 1) * sd_b**2) / df
        se = np.sqrt(pooled * (1 / n_a + 1 / n_b))
        mean_diff = mean_a - mean_b
        t_stat = mean_diff / se
        alpha = self.parameters.get("alpha", 0.05)
        alternative = self.parameters.get("alt_hypothesis", "two-sided")
        lower, upper = st.t.interval(1 - alpha, df, loc=mean_diff, scale=se)
        if alternative == "greater":
            p = 1.0 - st.t.cdf(t_stat, df)
            upper = "Infinity"
        elif alternative == "less":
            p = 1.0 - st.t.cdf(-t_stat, df)
            lower = "-Infinity"
        else:
            p = 2 * (1.0 - st.t.cdf(abs(t_stat), df))
        return TTestIndependentResult(
            t_stat=t_stat,
            df=df,
            p=p,
            mean_diff=mean_diff,
            se_diff=se,
            ci_upper=upper,
            ci_lower=lower,
            cohens_d=mean_diff / np.sqrt(pooled),
        )
