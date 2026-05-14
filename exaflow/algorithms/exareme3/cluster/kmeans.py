from typing import List

from pydantic import BaseModel

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.cluster.kmeans import FederatedKMeans


class KMeansResult(BaseModel):
    title: str
    n_obs: int
    centers: List[List[float]]


class KMeans(Algorithm):
    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="kmeans",
            desc="K-Means clustering for numerical variables.",
            documentation=(
                "Cluster selected numerical variables with K-Means using "
                "aggregation-server-backed initialization.\n\n"
                "The 'k' setting controls the number of clusters. Default is 4.\n\n"
                "The 'maxiter' setting controls the maximum number of fitting "
                "iterations. Default is 1.\n\n"
                "The 'tol' setting controls the convergence tolerance. Default "
                "is 0.01.\n\n"
                "The result includes cluster assignments and fitted cluster "
                "centers."
            ),
            label="Federated K-Means",
            enabled=True,
            inputdata=specs.InputDataSpecifications(
                y=specs.InputDataSpecification(
                    label="y",
                    desc="Numerical variables used for clustering.",
                    types=[specs.InputDataType.REAL, specs.InputDataType.INT],
                    stattypes=[specs.InputDataStatType.NUMERICAL],
                    required=True,
                    multiple=True,
                ),
            ),
            parameters={
                "k": specs.ParameterSpecification(
                    label="k",
                    desc="Number of clusters to fit.",
                    types=[specs.ParameterType.INT],
                    required=True,
                    multiple=False,
                    default=4,
                    min=1,
                    max=100,
                ),
                "maxiter": specs.ParameterSpecification(
                    label="maxiter",
                    desc="Maximum number of fitting iterations.",
                    types=[specs.ParameterType.INT],
                    required=True,
                    multiple=False,
                    default=1,
                    min=1,
                    max=100,
                ),
                "tol": specs.ParameterSpecification(
                    label="tol",
                    desc="Convergence tolerance for fitting.",
                    types=[specs.ParameterType.REAL],
                    required=True,
                    multiple=False,
                    default=0.01,
                    min=0.0,
                    max=1.0,
                ),
            },
            type=specs.AlgorithmType.EXAREME3,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def run(self):
        n_clusters = int(self.get_parameter("k"))

        tol = float(self.get_parameter("tol", 1e-4))
        maxiter = int(self.get_parameter("maxiter", 100))

        result = self.run_local_udf(
            func=local_step,
            kw_args={
                "n_clusters": n_clusters,
                "tol": tol,
                "maxiter": maxiter,
            },
            identical_results=True,
        )
        n_obs = int(result["n_obs"])
        centers = result["centers"]

        return KMeansResult(
            title="K-Means Centers",
            n_obs=n_obs,
            centers=centers,
        )


@exareme3_udf(with_aggregation_server=True)
def local_step(agg_client, data, n_clusters, tol, maxiter):
    estimator = FederatedKMeans(
        agg_client=agg_client,
        n_clusters=n_clusters,
        tol=tol,
        maxiter=maxiter,
    ).fit(data)

    return dict(n_obs=estimator.n_obs_, centers=estimator.cluster_centers_)
