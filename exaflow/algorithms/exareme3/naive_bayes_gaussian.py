from typing import List

from pydantic import BaseModel

from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.naive_bayes_gaussian import FederatedGaussianNB

ALGORITHM_NAME = "naive_bayes_gaussian"


class NaiveBayesGaussianResult(BaseModel):
    classes: List[str]
    class_count: List[float]
    class_prior: List[float]
    theta: List[List[float]]
    var: List[List[float]]
    feature_names: List[str]


class NaiveBayesGaussianAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self) -> NaiveBayesGaussianResult:
        y_var = self.inputdata.y[0]
        x_vars = list(self.inputdata.x)

        labels = sorted(self.metadata[y_var]["enumerations"].keys())

        udf_results = self.run_local_udf(
            func=naive_bayes_gaussian_local_step,
            kw_args={
                "y_var": y_var,
                "x_vars": x_vars,
                "labels": labels,
            },
        )

        params = udf_results[0]
        return NaiveBayesGaussianResult(**params)


@exareme3_udf(with_aggregation_server=True)
def naive_bayes_gaussian_local_step(
    agg_client,
    data,
    y_var,
    x_vars,
    labels,
):
    X = data[x_vars].to_numpy(dtype=float, copy=False)
    y = data[y_var].to_numpy()

    model = FederatedGaussianNB(
        x_vars=x_vars,
        labels=labels,
    )
    results = model.fit(X, y, agg_client=agg_client)

    return {
        "classes": list(results.labels),
        "class_count": results.class_count.astype(float).tolist(),
        "class_prior": results.class_prior.astype(float).tolist(),
        "theta": results.theta.tolist(),
        "var": results.var.tolist(),
        "feature_names": list(results.x_vars),
    }
