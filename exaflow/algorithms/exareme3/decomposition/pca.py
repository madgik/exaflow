from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.decomposition.pca_common import PCAResult
from exaflow.algorithms.exareme3.decomposition.pca_common import build_pca_result
from exaflow.algorithms.exareme3.decomposition.pca_common import model_result_payload
from exaflow.algorithms.exareme3.decomposition.pca_common import run_federated_pca
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf


class PCA(Algorithm):
    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="pca",
            desc="Principal component analysis for numerical variables.",
            documentation=(
                "Computes principal components for selected numerical "
                "variables from their covariance structure. PCA summarizes "
                "multivariate variation by producing eigenvalues and "
                "eigenvectors for orthogonal component directions.\n\n"
                "The result includes the observation count, eigenvalues, and "
                "eigenvectors.\n\n"
                "Reference behavior is aligned with covariance-based PCA "
                "methodology as exposed by scikit-learn PCA. The method "
                "computes the covariance quantities from aggregated sufficient "
                "statistics without sharing raw data."
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
            type=specs.AlgorithmType.EXAREME3,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def run(self):
        result = self.run_local_udf(
            func=local_step,
            kw_args={
                "y_vars": self.y,
            },
            identical_results=True,
        )
        return build_pca_result(
            payload=result,
            variables=self.y,
            pca_variant="pca",
        )


@exareme3_udf(with_aggregation_server=True)
def local_step(agg_client, data, y_vars):
    execution = run_federated_pca(
        agg_client=agg_client,
        data=data,
        variables=y_vars,
    )
    return model_result_payload(execution.model)
