import numpy as np
import scipy.stats as st

from exaflow.algorithms.exareme3.statistics.ttest_independent import TTestIndependent
from exaflow.algorithms.exareme3.statistics.ttest_independent import (
    TTestIndependentResult,
)
from exaflow.algorithms.feta import FedTensor
from exaflow.algorithms.feta import federated_numpy as fnp
from exaflow.algorithms.feta.algorithm import Algorithm
from exaflow.algorithms.specifications import AlgorithmType
from exaflow.worker_communication import BadUserInput


class FetaTTestIndependent(Algorithm):
    @classmethod
    def get_specification(cls):
        specification = TTestIndependent.get_specification()
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
