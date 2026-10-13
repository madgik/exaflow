import numpy as np
import scipy.stats as st

from exaflow.algorithms.exareme3.statistics.ttest_onesample import TTestOneSample
from exaflow.algorithms.exareme3.statistics.ttest_onesample import TTestOneSampleResult
from exaflow.algorithms.feta import FedTensor
from exaflow.algorithms.feta import federated_numpy as fnp
from exaflow.algorithms.feta.algorithm import Algorithm
from exaflow.algorithms.specifications import AlgorithmType
from exaflow.worker_communication import BadUserInput


class FetaTTestOneSample(Algorithm):
    @classmethod
    def get_specification(cls):
        specification = TTestOneSample.get_specification()
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
