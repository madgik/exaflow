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
            desc="Federated PCA that computes eigenvalues and eigenvectors from aggregated covariances to summarize multivariate structure.",
            label="Principal Component Analysis",
            enabled=True,
            inputdata=specs.InputDataSpecifications(
                y=specs.InputDataSpecification(
                    label="Variables (features)",
                    desc="List of numerical variables used to compute the covariance matrix.",
                    types=[specs.InputDataType.REAL, specs.InputDataType.INT],
                    stattypes=[specs.InputDataStatType.NUMERICAL],
                    required=True,
                    multiple=True,
                    enumslen=None,
                ),
                x=None,
                validation=None,
            ),
            parameters=None,
            type=specs.AlgorithmType.EXAREME3,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def run(self):
        results = self.run_local_udf(
            func=local_step,
            kw_args={
                "y_vars": self.inputdata.y,
            },
        )
        result = results[0]
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
