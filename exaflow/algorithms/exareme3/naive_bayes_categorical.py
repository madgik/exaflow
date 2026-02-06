from typing import Dict
from typing import List

from pydantic import BaseModel

from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.naive_bayes_categorical import FederatedCategoricalNB
from exaflow.algorithms.federated.preprocessing.ordinal_encoder import (
    FederatedOrdinalEncoder,
)

ALGORITHM_NAME = "naive_bayes_categorical"


class NaiveBayesCategoricalResult(BaseModel):
    classes: List[str]
    class_count: List[float]
    class_log_prior: List[float]
    category_count: Dict[str, List[List[float]]]
    category_log_prob: Dict[str, List[List[float]]]
    categories: Dict[str, List[str]]
    feature_names: List[str]


class NaiveBayesCategoricalAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self):
        y_var = self.inputdata.y[0]
        x_vars = list(self.inputdata.x)

        categories: Dict[str, List[str]] = {
            var: list(sorted(self.metadata[var]["enumerations"].keys()))
            for var in x_vars + [y_var]
        }

        udf_results = self.run_local_udf(
            func=naive_bayes_categorical_local_step,
            kw_args={
                "y_var": y_var,
                "x_vars": x_vars,
                "categories": categories,
            },
        )

        params = udf_results[0]
        return NaiveBayesCategoricalResult(**params)


@exareme3_udf(with_aggregation_server=True)
def naive_bayes_categorical_local_step(
    agg_client,
    data,
    y_var,
    x_vars,
    categories,
):
    encoder = FederatedOrdinalEncoder(
        categories=categories,
        handle_unknown="error",
    )
    encoder.fit(
        agg_client=agg_client,
        data=data,
        categorical_vars=x_vars,
    )
    X = encoder.transform(
        data,
        categorical_vars=x_vars,
        numerical_vars=[],
    )
    y = data[y_var].to_numpy()

    model = FederatedCategoricalNB(
        y_var=y_var,
        x_vars=x_vars,
        categories=categories,
    )
    results = model.fit(X, y, agg_client=agg_client)
    return {
        "classes": list(results.labels),
        "class_count": results.class_count.astype(float).tolist(),
        "class_log_prior": results.class_log_prior.tolist(),
        "category_count": {
            xvar: results.category_count.get(xvar, []).tolist()
            for xvar in results.x_vars
        },
        "category_log_prob": {
            xvar: results.category_log_prob.get(xvar, []).tolist()
            for xvar in results.x_vars
        },
        "categories": {
            xvar: list(results.categories.get(xvar, [])) for xvar in results.x_vars
        },
        "feature_names": list(results.x_vars),
    }
