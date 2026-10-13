import numpy as np
import scipy.stats as st

from exaflow.algorithms.exareme3.statistics.ttest_paired import TTestPaired
from exaflow.algorithms.exareme3.statistics.ttest_paired import TTestPairedResult
from exaflow.algorithms.feta import FedTensor
from exaflow.algorithms.feta import federated_numpy as fnp
from exaflow.algorithms.feta.algorithm import Algorithm
from exaflow.algorithms.specifications import AlgorithmType
from exaflow.worker_communication import BadUserInput


class FetaTTestPaired(Algorithm):
    @classmethod
    def get_specification(cls):
        specification = TTestPaired.get_specification()
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
