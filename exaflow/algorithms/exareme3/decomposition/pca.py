from typing import List

from pydantic import BaseModel

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.decomposition.pca import FederatedPCA


class PCAResult(BaseModel):
    title: str
    n_obs: int
    eigenvalues: List[float]
    eigenvectors: List[List[float]]


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
            inputdata=specs.InputDataSpecifications(
                y=specs.InputDataSpecification(
                    label="Variables",
                    desc="Numerical variables used to compute principal components.",
                    types=[specs.InputDataType.REAL, specs.InputDataType.INT],
                    stattypes=[specs.InputDataStatType.NUMERICAL],
                    required=True,
                ),
            ),
            type=specs.AlgorithmType.EXAREME3,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def run(self):
        result = self.run_local_udf(
            func=local_step,
            kw_args={
                "y_vars": self.inputdata.y,
            },
            identical_results=True,
        )
        n_obs = result["n_obs"]
        eigenvalues = result["eigenvalues"]
        eigenvectors = result["eigenvectors"]

        result = PCAResult(
            title="Eigenvalues and Eigenvectors",
            n_obs=n_obs,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
        )
        return result


@exareme3_udf(with_aggregation_server=True)
def local_step(agg_client, data, y_vars):
    X = data[y_vars]

    model = FederatedPCA(agg_client=agg_client)
    model.fit(X)
    return dict(
        n_obs=model.n_samples_seen_,
        eigenvalues=model.explained_variance_.tolist(),
        eigenvectors=model.components_.tolist(),
    )
