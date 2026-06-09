from typing import Dict
from typing import List

from pydantic import BaseModel

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.metadata_enums import get_enum_codes
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
            desc="Categorical Naive Bayes classification for nominal features.",
            documentation=(
                "Fits a categorical Naive Bayes classifier for a nominal "
                "outcome using nominal features. Features are ordinal-encoded "
                "using metadata category order; unknown categories are "
                "rejected.\n\n"
                "The result includes class labels, class counts, class log "
                "priors, per-feature category counts, per-feature category log "
                "probabilities, category labels, and feature names.\n\n"
                "Reference behavior is aligned with scikit-learn CategoricalNB "
                "methodology, using aggregated class and category counts "
                "without sharing raw data."
            ),
            label="Categorical Naive Bayes",
            enabled=True,
            required_preprocessing=["missing_values_handler"],
            inputdata=specs.InputDataSpecifications(
                y=specs.InputDataSpecification(
                    label="Outcome",
                    desc="Nominal outcome variable.",
                    types=[specs.InputDataType.TEXT],
                    stattypes=[specs.InputDataStatType.NOMINAL],
                    required=True,
                    max_count=1,
                ),
                x=specs.InputDataSpecification(
                    label="Features",
                    desc="Nominal features used for classification.",
                    types=[specs.InputDataType.TEXT],
                    stattypes=[specs.InputDataStatType.NOMINAL],
                    required=True,
                ),
            ),
            type=specs.AlgorithmType.EXAREME3,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def run(self):
        y_var = self.inputdata.y[0]
        x_vars = list(self.inputdata.x)
        categories: Dict[str, List[str]] = {
            var: sorted(get_enum_codes(self.metadata, var)) for var in x_vars + [y_var]
        }

        params = self.run_local_udf(
            func=local_step,
            kw_args={
                "y_var": y_var,
                "x_vars": x_vars,
                "categories": categories,
            },
            identical_results=True,
        )
        return NaiveBayesCategoricalResult(**params)


@exareme3_udf(with_aggregation_server=True)
def local_step(
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
