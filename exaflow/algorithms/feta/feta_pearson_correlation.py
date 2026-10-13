from typing import Sequence

import numpy as np
import scipy.stats as st
from pydantic import BaseModel
from scipy import special

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.feta import federated_numpy as fnp
from exaflow.algorithms.feta.algorithm import Algorithm
from exaflow.algorithms.feta.tensor import FedTensor
from exaflow.algorithms.specifications import AlgorithmType
from exaflow.worker_communication import BadUserInput


class PearsonResult(BaseModel):
    title: str
    n_obs: int
    correlations: dict
    p_values: dict
    ci_hi: dict
    ci_lo: dict


class FetaPearsonCorrelation(Algorithm):
    @classmethod
    def get_specification(cls):
        specification = specs.AlgorithmSpecification(
            name="feta_pearson_correlation",
            desc="Pearson correlations for selected numerical variables.",
            documentation=(
                "Compute Pearson correlations for all pairs between the primary "
                "variables and optional secondary variables. When no secondary "
                "variables are provided, correlations are computed among the "
                "primary variables. Missing values are handled by the selected "
                "preprocessing step before correlations are computed.\n\n"
                "The 'alpha' setting controls the confidence level for "
                "correlation coefficient intervals. Default is 0.95.\n\n"
                "The result includes correlation coefficients, p-values, "
                "confidence intervals, and observation counts.\n\n"
                "Reference behavior is aligned with standard Pearson product-"
                "moment correlation methodology as exposed by scipy.stats. "
                "The method computes the correlation matrix and inferential "
                "quantities from aggregated sufficient statistics without "
                "sharing raw data."
            ),
            label="Pearson Correlation",
            enabled=True,
            required_preprocessing=[],
            y=specs.InputDataSpecification(
                label="Variables",
                desc="Numerical variables for the primary correlation axis.",
                types=[specs.InputDataType.REAL, specs.InputDataType.INT],
                stattypes=[specs.InputDataStatType.NUMERICAL],
                required=True,
            ),
            x=specs.InputDataSpecification(
                label="Additional variables",
                desc="Optional numerical variables for the secondary axis.",
                types=[specs.InputDataType.REAL, specs.InputDataType.INT],
                stattypes=[specs.InputDataStatType.NUMERICAL],
                required=False,
            ),
            parameters={
                "alpha": specs.ParameterSpecification(
                    label="Confidence level",
                    desc="Confidence level for correlation intervals.",
                    types=[specs.ParameterType.REAL],
                    required=True,
                    multiple=False,
                    default=0.95,
                    min=0.0,
                    max=1.0,
                ),
            },
            type=specs.AlgorithmType.FETA,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )
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

    def run(self, y: FedTensor, x: FedTensor | None = None):
        if x is None:
            x = y
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
            result, row_names=self.x_names or self.y_names, column_names=self.y_names
        )
        return PearsonResult(
            title="Pearson Correlation Coefficient",
            n_obs=int(n),
            correlations=corr,
            p_values=p,
            ci_hi=hi,
            ci_lo=lo,
        )


def _format_result_matrices(
    result, *, row_names: Sequence[str], column_names: Sequence[str]
):
    correlations = result["correlations"]
    p_values = result["p_values"]
    ci_hi = result["ci_hi"]
    ci_lo = result["ci_lo"]

    def _build_matrix_dict(values_matrix):
        matrix_dict = {"variables": list(row_names)}
        matrix_dict.update({col: row for col, row in zip(column_names, values_matrix)})
        return matrix_dict

    return (
        _build_matrix_dict(correlations),
        _build_matrix_dict(p_values),
        _build_matrix_dict(ci_hi),
        _build_matrix_dict(ci_lo),
    )
