from typing import Dict
from typing import List

from pydantic import BaseModel

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.naive_bayes import FederatedCategoricalNB
from exaflow.algorithms.federated.pipeline import FederatedPipeline
from exaflow.algorithms.federated.preprocessing import FederatedOrdinalEncoder


class NaiveBayesCategoricalResult(BaseModel):
    classes: List[str]
    class_count: List[float]
    class_log_prior: List[float]
    category_count: Dict[str, List[List[float]]]
    category_log_prob: Dict[str, List[List[float]]]
    categories: Dict[str, List[str]]
    feature_names: List[str]


class NaiveBayesCategorical(Algorithm):
    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="naive_bayes_categorical",
            desc="Federated categorical Naive Bayes. Features are ordinal-encoded using metadata category order; unknown categories are rejected. Class labels are discovered during training and aggregated securely across workers.",
            label="Categorical Naive Bayes",
            enabled=True,
            inputdata=specs.InputDataSpecifications(
                y=specs.InputDataSpecification(
                    label="Variable (dependent)",
                    desc="A unique nominal variable.",
                    types=[specs.InputDataType.TEXT, specs.InputDataType.INT],
                    stattypes=[specs.InputDataStatType.NOMINAL],
                    required=True,
                    multiple=False,
                    enumslen=None,
                ),
                x=specs.InputDataSpecification(
                    label="Covariates (independent)",
                    desc="One or more nominal variables.",
                    types=[specs.InputDataType.TEXT, specs.InputDataType.INT],
                    stattypes=[specs.InputDataStatType.NOMINAL],
                    required=True,
                    multiple=True,
                    enumslen=None,
                ),
                validation=None,
            ),
            parameters=None,
            type=specs.AlgorithmType.EXAREME3,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

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
    y = data[y_var].to_numpy()

    pipeline = FederatedPipeline(
        [
            (
                "features",
                FederatedOrdinalEncoder(
                    categories=categories,
                    handle_unknown="error",
                ),
            ),
            (
                "model",
                FederatedCategoricalNB(
                    y_var=y_var,
                    x_vars=x_vars,
                    categories=categories,
                ),
            ),
        ]
    )
    results = pipeline.fit(
        agg_client=agg_client,
        data=data,
        y=y,
        categorical_vars=x_vars,
        numerical_vars=[],
    )
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
