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
            desc="K-means clustering for numerical variables.",
            documentation=(
                "Partitions observations into k clusters based on selected "
                "numerical variables. Cluster centers are initialized from the "
                "global feature ranges and refined with Lloyd iterations.\n\n"
                "The k parameter controls the number of clusters. The maxiter "
                "parameter controls the maximum number of fitting iterations. "
                "The tol parameter controls the convergence tolerance, based "
                "on the Frobenius norm of the cluster-center update.\n\n"
                "At each iteration, observations are assigned to the nearest "
                "cluster center using squared Euclidean distance. New centers "
                "are computed from aggregated cluster-wise sums and counts. "
                "Empty clusters are reset to the origin.\n\n"
                "Reference behavior is aligned with classical Lloyd K-means, "
                "as exposed by scikit-learn KMeans with algorithm='lloyd'. "
                "This method uses initialization from global feature ranges "
                "and updates centers from aggregated cluster-wise sums and "
                "counts without sharing raw data.\n\n"
                "The result includes the total number of observations used for "
                "fitting and the fitted cluster centers."
            ),
            label="K-means",
            enabled=True,
            required_preprocessing=["missing_values_handler"],
            inputdata=specs.InputDataSpecifications(
                y=specs.InputDataSpecification(
                    label="Variables",
                    desc="Numerical variables used for clustering.",
                    types=[specs.InputDataType.REAL, specs.InputDataType.INT],
                    stattypes=[specs.InputDataStatType.NUMERICAL],
                    required=True,
                ),
            ),
            parameters={
                "k": specs.ParameterSpecification(
                    label="Number of clusters",
                    desc="Number of clusters to fit.",
                    types=[specs.ParameterType.INT],
                    required=True,
                    multiple=False,
                    default=4,
                    min=1,
                    max=100,
                ),
                "maxiter": specs.ParameterSpecification(
                    label="Maximum iterations",
                    desc="Maximum number of fitting iterations.",
                    types=[specs.ParameterType.INT],
                    required=True,
                    multiple=False,
                    default=1,
                    min=1,
                    max=100,
                ),
                "tol": specs.ParameterSpecification(
                    label="Convergence tolerance",
                    desc="Tolerance used to decide convergence.",
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
