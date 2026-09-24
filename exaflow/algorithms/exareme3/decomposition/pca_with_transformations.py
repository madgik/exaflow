from typing import Dict

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.decomposition.pca_common import PCAResult
from exaflow.algorithms.exareme3.decomposition.pca_common import build_pca_result
from exaflow.algorithms.exareme3.decomposition.pca_common import model_result_payload
from exaflow.algorithms.exareme3.decomposition.pca_common import run_federated_pca
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.worker_communication import BadUserInput


class PCAWithTransformation(Algorithm):
    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="pca_with_transformation",
            desc="Principal component analysis after optional variable transformations.",
            documentation=(
                "Computes principal components for selected numerical variables "
                "after optional per-variable transformations. Principal "
                "components can represent the original variables with reduced "
                "dimensionality.\n\n"
                "The data_transformation parameter selects transformations "
                "such as log, exp, center, or standardize for selected "
                "variables. Centering and standardization use global means and "
                "standard deviations; log transformation requires positive "
                "values.\n\n"
                "The result includes the observation count, eigenvalues, and "
                "eigenvectors.\n\n"
                "Reference behavior is aligned with covariance-based PCA "
                "methodology as exposed by scikit-learn PCA, after applying "
                "the requested transformations. Covariance quantities are "
                "computed from aggregated sufficient statistics without "
                "sharing raw data."
            ),
            label="Principal Component Analysis",
            enabled=True,
            required_preprocessing=["missing_values_handler"],
            y=specs.InputDataSpecification(
                label="Variables",
                desc="Numerical variables used to compute principal components.",
                types=[specs.InputDataType.REAL, specs.InputDataType.INT],
                stattypes=[specs.InputDataStatType.NUMERICAL],
                required=True,
            ),
            parameters={
                "data_transformation": specs.ParameterSpecification(
                    label="Data transformation",
                    desc="Transformation applied to each selected variable.",
                    types=[specs.ParameterType.DICT],
                    required=False,
                    multiple=False,
                ),
            },
            type=specs.AlgorithmType.EXAREME3,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def run(self):
        data_transformation: Dict = self.get_parameter("data_transformation")

        try:
            result = self.run_local_udf(
                func=local_step,
                kw_args={
                    "y_vars": self.y,
                    "data_transformation": data_transformation,
                },
                identical_results=True,
            )
        except Exception as ex:
            msg = str(ex)
            if (
                "Log transformation cannot be applied to non-positive values in column"
                in msg
                or "Unknown transformation" in msg
                or "Standardization cannot be applied to column" in msg
            ):
                raise BadUserInput(msg)
            raise

        return build_pca_result(
            payload=result,
            variables=self.y,
            pca_variant="pca_with_transformation",
            data_transformation=data_transformation,
        )


@exareme3_udf(with_aggregation_server=True)
def local_step(
    agg_client,
    data,
    y_vars,
    data_transformation,
):
    execution = run_federated_pca(
        agg_client=agg_client,
        data=data,
        variables=y_vars,
        data_transformation=data_transformation,
    )
    return model_result_payload(execution.model)
