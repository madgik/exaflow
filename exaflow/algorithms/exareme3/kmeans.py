from typing import List

from pydantic import BaseModel

from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.kmeans import FederatedKMeans

ALGORITHM_NAME = "kmeans"


class KMeansResult(BaseModel):
    title: str
    n_obs: int
    centers: List[List[float]]


class KMeansAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self):
        n_clusters = int(self.get_parameter("k"))

        tol = float(self.get_parameter("tol", 1e-4))
        maxiter = int(self.get_parameter("maxiter", 100))

        results = self.run_local_udf(
            func=local_step,
            kw_args={
                "n_clusters": n_clusters,
                "tol": tol,
                "maxiter": maxiter,
            },
        )

        result = results[0]
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
