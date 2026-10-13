import numpy as np
import scipy.stats as st
from scipy import special

from exaflow.algorithms.exareme3.statistics.pearson_correlation import (
    PearsonCorrelation,
)
from exaflow.algorithms.exareme3.statistics.pearson_correlation import PearsonResult
from exaflow.algorithms.exareme3.statistics.pearson_correlation import (
    _format_result_matrices,
)
from exaflow.algorithms.feta import federated_numpy as fnp
from exaflow.algorithms.feta.algorithm import Algorithm
from exaflow.algorithms.feta.tensor import FedTensor
from exaflow.algorithms.specifications import AlgorithmType
from exaflow.worker_communication import BadUserInput


class FetaPearsonCorrelation(Algorithm):
    @classmethod
    def get_specification(cls):
        specification = PearsonCorrelation.get_specification()
        documentation = specification.documentation.replace(
            "Missing values are handled by the selected preprocessing step before "
            "correlations are computed.",
            "Inputs containing missing values are rejected. Preprocessing is not supported.",
        )
        return specification.model_copy(
            update={
                "name": "feta_pearson_correlation",
                "type": AlgorithmType.FETA,
                "required_preprocessing": [],
                "documentation": documentation,
            }
        )

    @classmethod
    def resolve_inputs(cls, x, y):
        return list(x or y or []), list(y or [])

    def run(self, x: FedTensor, y: FedTensor):
        n = float(len(x))
        sx = fnp.sum(x, axis=0)
        sy = fnp.sum(y, axis=0)
        sxx = fnp.sum(x * x, axis=0)
        syy = fnp.sum(y * y, axis=0)
        sxy = fnp.transpose(y) @ x

        df = n - 2
        if n == 0:
            raise BadUserInput("Cannot compute Pearson correlation on empty data.")
        if df <= 0:
            raise BadUserInput(
                "Not enough observations to compute Pearson correlation."
            )
        alpha = self.parameters.get("alpha", 0.95)
        d = np.sqrt(n * sxx - sx * sx) * np.sqrt(n * syy - sy * sy)[:, np.newaxis]
        correlations = (n * sxy - sx * sy[:, np.newaxis]) / d
        correlations[d == 0] = 0
        correlations = correlations.clip(-1, 1)
        t_squared = correlations**2 * (
            df / ((1.0 - correlations) * (1.0 + correlations))
        )
        p_values = special.betainc(
            0.5 * df, 0.5, np.fmin(np.asarray(df / (df + t_squared)), 1.0)
        )
        p_values[abs(correlations) == 1] = 0
        r_z = np.arctanh(correlations)
        se = 1 / np.sqrt(n - 3)
        z = st.norm.ppf(1 - alpha / 2)
        ci_lo, ci_hi = np.tanh((r_z - z * se, r_z + z * se))
        result = {
            "correlations": correlations.tolist(),
            "p_values": p_values.tolist(),
            "ci_hi": ci_hi.tolist(),
            "ci_lo": ci_lo.tolist(),
        }
        corr, p, hi, lo = _format_result_matrices(
            result, row_names=self.x_names, column_names=self.y_names
        )
        return PearsonResult(
            title="Pearson Correlation Coefficient",
            n_obs=int(n),
            correlations=corr,
            p_values=p,
            ci_hi=hi,
            ci_lo=lo,
        )
